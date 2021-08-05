import copy
from os import path as osp

import numpy as np
import torch
from mmdet.datasets.builder import DATASETS

from mmocr.core import compute_f1_score
from mmocr.datasets.base_dataset import BaseDataset
from mmocr.datasets.pipelines import sort_vertex8
from mmocr.utils import is_type_list, list_from_file

from sklearn.metrics import precision_score, recall_score


@DATASETS.register_module()
class KIEDataset(BaseDataset):
    """
    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        loader (dict): Dictionary to construct loader
            to load annotation infos.
        img_prefix (str, optional): Image prefix to generate full
            image path.
        test_mode (bool, optional): If True, try...except will
            be turned off in __getitem__.
        dict_file (str): Character dict file path.
        norm (float): Norm to map value from one range to another.
    """

    def __init__(self,
                 ann_file,
                 loader,
                 dict_file,
                 img_prefix='',
                 pipeline=None,
                 norm=10.,
                 directed=False,
                 test_mode=True,
                 **kwargs):
        super().__init__(
            ann_file,
            loader,
            pipeline,
            img_prefix=img_prefix,
            test_mode=test_mode)
        assert osp.exists(dict_file)

        self.norm = norm
        self.directed = directed
        self.dict = {
            '': 0,
            **{
                line.rstrip('\r\n'): ind
                for ind, line in enumerate(list_from_file(dict_file), 1)
            }
        }

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['bbox_fields'] = []
        results['ori_texts'] = results['ann_info']['texts']
        results['filename'] = osp.join(self.img_prefix,
                                       results['img_info']['filename'])
        results['ori_filename'] = results['img_info']['filename']
        # a dummy img data
        results['img'] = np.zeros((0, 0, 0), dtype=np.uint8)

    def _parse_anno_info(self, annotations, pre_defined_edges=None):
        """Parse annotations of boxes, texts and labels for one image.
        Args:
            annotations (list[dict]): Annotations of one image, where
                each dict is for one character.

        Returns:
            dict: A dict containing the following keys:

                - bboxes (np.ndarray): Bbox in one image with shape:
                    box_num * 4. They are sorted clockwise when loading.
                - relations (np.ndarray): Relations between bbox with shape:
                    box_num * box_num * D.
                - texts (np.ndarray): Text index with shape:
                    box_num * text_max_len.
                - labels (np.ndarray): Box Labels with shape:
                    box_num * (box_num + 1).
        """

        assert is_type_list(annotations, dict)
        assert len(annotations) > 0, 'Please remove data with empty annotation'
        assert 'box' in annotations[0]
        assert 'text' in annotations[0]

        boxes, texts, text_inds, labels, edges = [], [], [], [], []
        for ann in annotations:
            box = ann['box']
            sorted_box = sort_vertex8(box[:8])
            boxes.append(sorted_box)
            text = ann['text']
            texts.append(ann['text'])
            text_ind = [self.dict[c] for c in text if c in self.dict]
            text_inds.append(text_ind)
            labels.append(ann.get('label', 0))
            edges.append(ann.get('edge', 0))

        ann_infos = dict(
            boxes=boxes,
            texts=texts,
            text_inds=text_inds,
            edges=edges,
            labels=labels)

        return self.list_to_numpy(ann_infos, pre_defined_edges)

    def prepare_train_img(self, index):
        """Get training data and annotations from pipeline.

        Args:
            index (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        img_ann_info = self.data_infos[index]
        img_info = {
            'filename': img_ann_info['file_name'],
            'height': img_ann_info['height'],
            'width': img_ann_info['width'],
        }
        pre_defined_edges = img_ann_info.get('edges', None)
        ann_info = self._parse_anno_info(img_ann_info['annotations'], pre_defined_edges=pre_defined_edges)
        results = dict(img_info=img_info, ann_info=ann_info)

        self.pre_pipeline(results)

        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metric='macro_f1',
                 metric_options=dict(macro_f1=dict(ignores=[])),
                 **kwargs):
        # allow some kwargs to pass through
        assert set(kwargs).issubset(['logger'])

        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['macro_f1']
        for m in metrics:
            if m not in allowed_metrics:
                raise KeyError(f'metric {m} is not supported')

        return self.compute_macro_f1(results, **metric_options['macro_f1'])

    def compute_macro_f1(self, results, ignores=[]):
        node_preds = []
        node_gts = []
        edge_preds = []
        edge_gts = []
        for idx, result in enumerate(results):
            node_preds.append(result['nodes'].cpu())
            box_ann_infos = self.data_infos[idx]['annotations']
            node_gt = [box_ann_info['label'] for box_ann_info in box_ann_infos]
            node_gts.append(torch.Tensor(node_gt))

            edge_preds.append(result['edges'].cpu())
            pre_defined_edges = self.data_infos[idx]['edges']
            edge_gt = np.zeros((len(box_ann_infos), len(box_ann_infos)), dtype=np.int32)
            edge_gt[np.array(pre_defined_edges)[:, 0], np.array(pre_defined_edges)[:, 1]] = 1
            np.fill_diagonal(edge_gt, -1)
            edge_gts.append(torch.Tensor(edge_gt).contiguous().view(-1))

        node_preds = torch.cat(node_preds)
        node_gts = torch.cat(node_gts).int()
        edge_preds = torch.cat(edge_preds)
        edge_gts = torch.cat(edge_gts).long()

        edge_valids = torch.nonzero(edge_gts != -1, as_tuple=False).view(-1)
        # acc_edge = (edge_preds[edge_valids].argmax(-1) == edge_gts[edge_valids]).numpy().mean()
        rec_edge = recall_score(edge_gts[edge_valids].numpy(), edge_preds[edge_valids].argmax(-1).numpy())
        pre_edge = precision_score(edge_gts[edge_valids].numpy(), edge_preds[edge_valids].argmax(-1).numpy())

        node_f1s = compute_f1_score(node_preds, node_gts, ignores)

        return {
            'macro_f1': node_f1s.mean(),
            'edge_precision': pre_edge,
            'edge_recall': rec_edge,
        }

    def list_to_numpy(self, ann_infos, pre_defined_edges=None):
        """Convert bboxes, relations, texts and labels to ndarray."""
        boxes, text_inds = ann_infos['boxes'], ann_infos['text_inds']
        boxes = np.array(boxes, np.int32)
        relations, bboxes = self.compute_relation(boxes)

        labels = ann_infos.get('labels', None)
        if labels is not None:
            labels = np.array(labels, np.int32)
            edges = ann_infos.get('edges', None)
            edges = None if pre_defined_edges is not None else edges
            if edges is not None:
                labels = labels[:, None]
                edges = np.array(edges)
                edges = (edges[:, None] == edges[None, :]).astype(np.int32)
                if self.directed:
                    edges = (edges & labels == 1).astype(np.int32)
                np.fill_diagonal(edges, -1)
                labels = np.concatenate([labels, edges], -1)
            elif pre_defined_edges is not None:
                labels = labels[:, None]
                edges = np.zeros((len(labels), len(labels)), dtype=np.int32)
                edges[np.array(pre_defined_edges)[:, 0], np.array(pre_defined_edges)[:, 1]] = 1
                if self.directed:
                    edges = (edges & labels == 1).astype(np.int32)
                np.fill_diagonal(edges, -1)
                labels = np.concatenate([labels, edges], -1)

        padded_text_inds = self.pad_text_indices(text_inds)

        return dict(
            bboxes=bboxes,
            relations=relations,
            texts=padded_text_inds,
            labels=labels)

    def pad_text_indices(self, text_inds):
        """Pad text index to same length."""
        max_len = max([len(text_ind) for text_ind in text_inds])
        padded_text_inds = -np.ones((len(text_inds), max_len), np.int32)
        for idx, text_ind in enumerate(text_inds):
            padded_text_inds[idx, :len(text_ind)] = np.array(text_ind)
        return padded_text_inds

    def compute_relation(self, boxes):
        """Compute relation between every two boxes."""
        x1, y1 = boxes[:, 0:1], boxes[:, 1:2]
        x2, y2 = boxes[:, 4:5], boxes[:, 5:6]
        w, h = np.maximum(x2 - x1 + 1, 1), np.maximum(y2 - y1 + 1, 1)
        dx = (x1.T - x1) / self.norm
        dy = (y1.T - y1) / self.norm
        xhh, xwh = h.T / h, w.T / h
        whs = w / h + np.zeros_like(xhh)
        relation = np.stack([dx, dy, whs, xhh, xwh], -1).astype(np.float32)
        bboxes = np.concatenate([x1, y1, x2, y2], -1).astype(np.float32)
        return relation, bboxes
