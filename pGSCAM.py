import numpy as np
import torch
from utils.ply import read_ply, write_ply
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import open3d as o3d


class PowerCAM:
    """
    Vanilla gradient class activation mapping
    """

    def __init__(
        self,
        input_model,
        batch,
        config,
        mask_type="none",
        mode="normal",
        norm=False,
        cls=-1,
    ):
        # mode: [normal, counterfactual]
        # mask_type: [none, single, subset]:- none(no mask), single(only single point), subset(collection of points)

        self.input_model = input_model
        self.batch = batch
        self.cls = cls
        self.config = config
        self.norm = norm
        self.is_masked = True
        self.threshold = [0.1, 0.3, 0.3]
        self.mode = mode
        self.mask_type = mask_type

        # actuall labels starts from unlabelled but we are ignoring it
        self.transform_map = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}

    def create_mask(self):
        # logits: [1, d, N]
        # points: [1, N, 3]
        # preds: [1, N]
        logits = self.logits
        softmax = torch.nn.Softmax(1)
        preds = softmax(logits).argmax(dim=1)
        point_0 = self.xyzn
        if self.mask_type == "none":
            logits_mask = torch.ones_like(logits)

        elif self.mask_type == "subset":
            # Create Mask
            pred_mask = (preds == self.cls).int()
            inds_mask = torch.argwhere(pred_mask.squeeze()).squeeze()
            masked_point = point_0[:, inds_mask, :].detach().cpu().numpy()

            # Perform DBSCAN clustering to seperate entities of class self.cls
            clustering = DBSCAN(
                eps=0.5, min_samples=5, metric="euclidean", algorithm="auto"
            ).fit(masked_point.squeeze())

            cluster_labels = clustering.labels_

            # Let's extract cluster 0
            inds_clust = np.argwhere((cluster_labels == 0).astype(int)).squeeze()
            clust_point = point_0[:, inds_mask[inds_clust], :]
            logits_mask = torch.zeros_like(logits)
            logits_mask[:, :, inds_mask[inds_clust]] = 1

            preds_mask = -torch.ones_like(preds)
            preds_mask[:, inds_mask[inds_clust]] = 1

            entities = [
                point_0[0].detach().cpu().numpy(),
                preds_mask[0].cpu().numpy().astype(np.int32),
            ]

            write_ply("vis_cluster.ply", entities, ["x", "y", "z", "preds"])

        elif self.mask_type == "single":
            # Create Mask
            pred_mask = (preds == self.cls).int()
            inds_mask = torch.argwhere(pred_mask.squeeze()).squeeze()

            # First element of inds mask as ROI
            logits_mask = torch.zeros_like(logits)

            # Logits
            logits_mask[:, :, inds_mask[0]] = 1

        return logits_mask

    #         return inds_mask[inds_clust]

    def getGradients(self):
        #         print(f"Current class: {self.transform_map[self.cls]}")
        self.input_model.eval()
        self.x, self.logits, self.activations, self.xyzn = self.input_model(self.batch)
        logits = self.logits
        #         logits = self.create_mask()*logits
        softmax = torch.nn.Softmax(1)
        preds = softmax(logits).argmax(dim=1)

        #         logits = logits[:, :, self.create_mask()]
        logits = self.create_mask() * logits

        logits = softmax(logits)
        mask = ((preds.squeeze(0) == self.cls).unsqueeze(0)).unsqueeze(1)
        #         print(f"Number of points for class {self.transform_map[self.cls]}: ", torch.sum(mask.squeeze()).item())

        logits = logits[:, self.cls, :]
        logits = torch.sum(logits, axis=-1)

        logits = logits.squeeze()

        self.logits = logits

        self.logits.backward(retain_graph=True)

        return torch.sum(mask.squeeze()).item()

    def heatmap(self):
        #         self.logits.backward(retain_graph=True)
        #         self.logits.backward()

        heatmaps_III = []
        heatmaps_III_kdtree = []
        point_0 = self.xyzn.cpu().numpy().squeeze()
        #         tree = KDTree(point_0.cpu().numpy().squeeze())
        #         print(point_0.shape)

        for i, act in enumerate(self.activations):
            grads = act.grad
            if self.mode == "normal":
                alpha = torch.sum(grads, axis=(0, 2))  # replaced from (2,3)
            elif self.mode == "counterfactual":
                alpha = -torch.sum(grads, axis=(0, 2))  # replaced from (2,3)
            activation = act.squeeze()
            heatmap = torch.matmul(alpha, activation)

            # Apply ReLU
            heatmap = torch.maximum(heatmap, torch.zeros_like(heatmap))

            # Normalize
            max_val = torch.max(heatmap, dim=-1, keepdim=True)[0]
            min_val = torch.min(heatmap, dim=-1, keepdim=True)[0]
            heatmap = (heatmap - min_val) / (max_val - min_val)
            heatmap = heatmap.cpu().detach().numpy()

            # Fill NaN values
            heatmap = np.nan_to_num(heatmap)

            heatmaps_III.append(heatmap)

            heatmap = heatmap.squeeze()

            if act.shape[2] != point_0.shape[1]:
                for pt in self.end_points["xyz"]:
                    if pt.shape[1] == act.shape[2]:
                        tree = KDTree(pt.cpu().numpy().squeeze(), leaf_size=40)
                        idx = tree.query(point_0, return_distance=False).squeeze()
                        heatmaps_III_kdtree.append(np.expand_dims(heatmap[idx], 0))
            else:
                heatmaps_III_kdtree.append(np.expand_dims(heatmap[i], 0))
        #             print(act.shape, heatmap.shape)
        self.heatmaps_III = heatmaps_III
        self.heatmaps_III_kdtree = heatmaps_III_kdtree
        pts = np.array(point_0.T[:, :3])
        print(type(pts), type(np.array(self.heatmaps_III)[3]))
        visualize_pointcloud_heatmap(pts, np.array(self.heatmaps_III)[3])

    def refinement(self):
        hm = self.heatmaps_III_kdtree[-1].squeeze()
        logits = self.end_points["logits"]
        #         logits = self.create_mask()*logits
        softmax = torch.nn.Softmax(1)
        preds = softmax(logits).argmax(dim=1).squeeze().detach().cpu().numpy()

        preds_mask = (preds == self.cls).astype(np.int32)

        hm_mask = (hm > 0.5).astype(np.int32)

        pred_final = hm_mask * preds_mask

        print(pred_final.shape, np.unique(hm_mask), np.unique(preds_mask))

    def visCAM(self):
        print("Saving visuals...")
        points = self.end_points["xyz"]
        labels = self.end_points["labels"].cpu().numpy().astype(np.int32)
        preds = self.end_points["logits"].cpu()

        for hm_i, hm in enumerate(self.heatmaps_III):
            for p_i, p in enumerate(points):
                if hm.shape[1] == p.shape[1]:
                    entities = [p[0].detach().cpu().numpy(), hm[0]]
                    break

            write_ply(
                f"./visuals/{self.mask_type}_{self.mode}_{self.transform_map[self.cls]}_{hm_i}_pgscam.ply",
                entities,
                ["x", "y", "z", "heatmap"],
            )

        for hm_i, hm in enumerate(self.heatmaps_III_kdtree):
            for p_i, p in enumerate(points):
                if hm.shape[1] == p.shape[1]:
                    entities = [p[0].detach().cpu().numpy(), hm[0]]
                    break

            write_ply(
                f"./visuals/{self.mask_type}_{self.mode}_{self.transform_map[self.cls]}_{hm_i}_pgscam_kdtree.ply",
                entities,
                ["x", "y", "z", "heatmap"],
            )

    #     def visCAM(self):
    #         print("Saving visuals...")
    #         points = self.end_points['xyz']  # list
    #         labels = self.end_points['labels'].cpu().numpy().astype(np.int32)  # [1, N]
    #         preds = self.end_points['logits'].cpu()
    #         softmax = torch.nn.Softmax(1)
    #         preds = softmax(preds).argmax(dim=1).detach().numpy().astype(np.int32) # [1, N]

    #         for hm_i, hm in enumerate(self.heatmaps_I_II_III):
    #             for p_i, p in enumerate(points):
    #                 if hm.shape[1] == p.shape[1]:
    #                     entities = [p[0].detach().cpu().numpy(), hm[0].detach().cpu().numpy()]
    #             if hm.shape[1] == points[0].shape[1]:
    #                 entities += [labels[0], preds[0]]
    #                 write_ply(f'./visuals/{self.transform_map[self.cls]}_{hm_i}_vanillaCam.ply', entities, ['x', 'y', 'z', 'heatmap', 'labels', 'preds'])
    #                 continue
    #             write_ply(f'./visuals/{self.transform_map[self.cls]}_{hm_i}_vanillaCam.ply', entities, ['x', 'y', 'z', 'heatmap'])

    def runCAM(self):
        num_points = self.getGradients()
        self.heatmap()
        return num_points


class IoUCalculator_heatmaps:
    def __init__(self):
        self.num_classes = 2
        self.gt_classes = [0 for _ in range(self.num_classes)]
        self.positive_classes = [0 for _ in range(self.num_classes)]
        self.true_positive_classes = [0 for _ in range(self.num_classes)]

    def add_data(self, hm, pred_mask):
        hm_valid = hm.detach().cpu().numpy()
        pred_mask_valid = pred_mask.detach().cpu().numpy()

        val_total_correct = 0
        val_total_seen = 0

        correct = np.sum(hm_valid == pred_mask_valid)
        val_total_correct += correct
        val_total_seen += len(pred_mask_valid)

        conf_matrix = confusion_matrix(
            pred_mask_valid, hm_valid, labels=np.arange(0, self.num_classes, 1)
        )
        self.gt_classes += np.sum(conf_matrix, axis=1)
        self.positive_classes += np.sum(conf_matrix, axis=0)
        self.true_positive_classes += np.diagonal(conf_matrix)

    def compute_iou(self):
        iou_list = []
        for n in range(0, self.num_classes, 1):
            if (
                float(
                    self.gt_classes[n]
                    + self.positive_classes[n]
                    - self.true_positive_classes[n]
                )
                != 0
            ):
                iou = self.true_positive_classes[n] / float(
                    self.gt_classes[n]
                    + self.positive_classes[n]
                    - self.true_positive_classes[n]
                )
                iou_list.append(iou)
            else:
                iou_list.append(0.0)
        mean_iou = sum(iou_list) / float(self.num_classes)
        return mean_iou, iou_list

    # Function to visualize the heatmap of the point cloud


def visualize_pointcloud_heatmap(pts, values, colormap="viridis"):
    if not isinstance(pts, np.ndarray):
        raise TypeError(f"Expected pts to be a numpy array, but got {type(pts)}")
    if not isinstance(values, np.ndarray):
        raise TypeError(f"Expected values to be a numpy array, but got {type(values)}")

    # Normalize heatmap values to range [0, 1]
    values_normalized = (values - np.min(values)) / (np.max(values) - np.min(values))

    # Convert normalized heatmap values to RGB colors using the specified colormap
    cmap = plt.get_cmap(colormap)
    colors = cmap(values_normalized)[:, :3]  # Ignore alpha channel

    # Create an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd], window_name="3D Point Cloud Heatmap")
