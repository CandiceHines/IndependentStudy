# SpatialProteomics APIs — names only

## spatialproteomics/__init__.py
(No public functions/classes)

## spatialproteomics/base_logger.py
(No public functions/classes)

## spatialproteomics/constants.py
Classes:
- Layers
- Dims
- Attrs
- Props
- Features
- Labels
- SDLayers
- SDFeatures

## spatialproteomics/container.py
Functions:
- load_image_data
- read_from_spatialdata

## spatialproteomics/sd/__init__.py
(No public functions/classes)

## spatialproteomics/sd/utils.py
(No public functions/classes)

## spatialproteomics/pp/__init__.py
(No public functions/classes)

## spatialproteomics/pp/intensity.py
Functions:
- is_positive
- percentage_positive

## spatialproteomics/pp/preprocessing.py
Functions:
- add_quantification
- add_observations
- apply
- threshold
- transform_expression_matrix
- filter_by_obs
- grow_cells
Classes:
- PreprocessingAccessor
  - get_bbox
  - get_channels
  - add_channel
  - add_segmentation
  - add_layer
  - add_layer_from_dataframe
  - add_observations
  - drop_observations
  - add_feature
  - add_obs_from_dataframe
  - add_quantification
  - add_quantification_from_dataframe
  - drop_layers
  - threshold
  - apply
  - normalize
  - downsample
  - rescale
  - filter_by_obs
  - remove_outlying_cells
  - grow_cells
  - merge_segmentation
  - get_layer_as_df
  - get_disconnected_cell
  - transform_expression_matrix
  - mask_region
  - mask_cells
  - convert_to_8bit

## spatialproteomics/pp/utils.py
Functions:
- merge
- handle_disconnected_cells

## spatialproteomics/la/__init__.py
(No public functions/classes)

## spatialproteomics/la/label.py
Functions:
- threshold_labels
- predict_cell_types_argmax
- predict_cell_subtypes
Classes:
- LabelAccessor
  - deselect
  - add_label_type
  - remove_label_type
  - add_label_property
  - set_label_name
  - set_label_colors
  - predict_cell_types_argmax
  - threshold_labels
  - add_labels
  - add_labels_from_dataframe
  - add_properties
  - predict_cell_subtypes
  - set_label_level

## spatialproteomics/la/utils.py
(No public functions/classes)

## spatialproteomics/nh/__init__.py
(No public functions/classes)

## spatialproteomics/nh/neighborhood.py
Classes:
- NeighborhoodAccessor
  - deselect
  - add_properties
  - add_neighborhoods_from_dataframe
  - set_neighborhood_colors
  - set_neighborhood_name
  - compute_neighborhoods_radius
  - compute_neighborhoods_knn
  - compute_neighborhoods_delaunay
  - add_neighborhood_obs
  - compute_graph_features

## spatialproteomics/nh/utils.py
(No public functions/classes)

## spatialproteomics/pl/__init__.py
(No public functions/classes)

## spatialproteomics/pl/plot.py
Classes:
- PlotAccessor
  - colorize
  - show
  - annotate
  - render_segmentation
  - render_labels
  - render_neighborhoods
  - render_obs
  - imshow
  - scatter_labels
  - scatter
  - add_box
  - autocrop

## spatialproteomics/pl/utils.py
(No public functions/classes)

## spatialproteomics/tl/__init__.py
(No public functions/classes)

## spatialproteomics/tl/tool.py
Functions:
- cellpose
- stardist
- mesmer
- astir
Classes:
- ToolAccessor
  - cellpose
  - stardist
  - mesmer
  - astir
  - convert_to_anndata
  - convert_to_spatialdata

## spatialproteomics/tl/utils.py
(No public functions/classes)

## spatialproteomics/image_container/ImageContainer.py
Classes:
- ImageContainer
  - compute_neighborhoods
  - get_neighborhood_composition

## spatialproteomics/image_container/__init__.py
(No public functions/classes)
