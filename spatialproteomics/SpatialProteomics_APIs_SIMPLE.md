# SpatialProteomics: file-by-file function & method list

## spatialproteomics/__init__.py
- (No public functions or classes found)

## spatialproteomics/base_logger.py
- (No public functions or classes found)

## spatialproteomics/constants.py
Classes:
- **Layers** — No docstring.
- **Dims** — No docstring.
- **Attrs** — No docstring.
- **Props** — No docstring.
- **Features** — No docstring.
- **Labels** — No docstring.
- **SDLayers** — No docstring.
- **SDFeatures** — No docstring.

## spatialproteomics/container.py
Functions:
- **load_image_data** — Creates a image container.
- **read_from_spatialdata** — Read data from a spatialdata object into the spatialproteomics object.

## spatialproteomics/image_container/ImageContainer.py
Classes:
- **ImageContainer** — This class is used to store multiple SpatialProteomics objects and perform operations on them.
  - compute_neighborhoods — Compute neighborhoods for spatial proteomics objects using the specified method and perform clustering.
  - get_neighborhood_composition — Get the composition of neighborhoods across all objects in the ImageContainer.

## spatialproteomics/image_container/__init__.py
- (No public functions or classes found)

## spatialproteomics/la/__init__.py
- (No public functions or classes found)

## spatialproteomics/la/label.py
Functions:
- **threshold_labels** — Binarise based on a threshold.
- **predict_cell_types_argmax** — This function predicts cell types based on the expression matrix using the argmax method.
- **predict_cell_subtypes** — This function predicts cell subtypes based on the expression matrix using a subtype dictionary.
Classes:
- **LabelAccessor** — Adds functions for cell phenotyping.
  - deselect — Deselect specific label indices from the data object.
  - add_label_type — Add a new label type to the data object.
  - remove_label_type — Remove specific cell type label(s) from the data object.
  - add_label_property — Add a label property for each unique cell type label.
  - set_label_name — Set the name of a specific cell type label.
  - set_label_colors — Set the color of a specific cell type label.
  - predict_cell_types_argmax — Predicts cell types based on the argmax classification of marker intensities.
  - threshold_labels — Binarise based on a threshold.
  - add_labels — Add labels from a mapping (cell -> label) to the spatialproteomics object.
  - add_labels_from_dataframe — Adds labels to the image container.
  - add_properties — Adds properties to the image container.
  - predict_cell_subtypes — Predict cell subtypes based on the binarized marker intensities.
  - set_label_level — Set the label level to a specific level.

## spatialproteomics/la/utils.py
- (No public functions or classes found)

## spatialproteomics/nh/__init__.py
- (No public functions or classes found)

## spatialproteomics/nh/neighborhood.py
Classes:
- **NeighborhoodAccessor** — Adds functions for cellular neighborhoods.
  - deselect — Deselect specific neighborhood indices from the data object.
  - add_properties — Adds neighborhood properties to the image container.
  - add_neighborhoods_from_dataframe — Add neighborhoods to the dataset from a DataFrame.
  - set_neighborhood_colors — Set the color of a specific neighborhood.
  - set_neighborhood_name — Set the name of one or more neighborhoods.
  - compute_neighborhoods_radius — Compute the neighborhoods of each cell based on a specified radius.
  - compute_neighborhoods_knn — Compute the neighborhoods of each cell based on k-nearest neighbors.
  - compute_neighborhoods_delaunay — Compute the neighborhoods of each cell based on a Delaunay triangulation.
  - add_neighborhood_obs — Adds neighborhood observations to the object by computing network features from the adjacency matrix.
  - compute_graph_features — Compute various graph features from the adjacency matrix of the data.

## spatialproteomics/nh/utils.py
- (No public functions or classes found)

## spatialproteomics/pl/__init__.py
- (No public functions or classes found)

## spatialproteomics/pl/plot.py
Classes:
- **PlotAccessor** — Adds plotting functions to the image container.
  - colorize — Colorizes a stack of images.
  - show — Display an image with optional rendering elements.
  - annotate — Annotates cells with their respective number on the plot.
  - render_segmentation — Renders the segmentation mask with optional alpha blending and boundary rendering.
  - render_labels — Renders cell type labels on the plot.
  - render_neighborhoods — Render neighborhoods on the spatial data.
  - render_obs — Render the observation layer with the specified feature and colormap.
  - imshow — Plots the image after rendering certain layers.
  - scatter_labels — Scatter plot of labeled cells.
  - scatter — Create a scatter plot of some feature.
  - add_box — Adds a rectangular box to the current plot.
  - autocrop — Crop the image so that the background surrounding the tissue/TMA gets cropped away.

## spatialproteomics/pl/utils.py
- (No public functions or classes found)

## spatialproteomics/pp/__init__.py
- (No public functions or classes found)

## spatialproteomics/pp/intensity.py
Functions:
- **is_positive** — Determines whether a cell is positive based on the provided intensity image and threshold.
- **percentage_positive** — Computes the percentage of positive pixels per label on the provided intensity image and region mask.

## spatialproteomics/pp/preprocessing.py
Functions:
- **add_quantification** — This function computes the quantification of the image data based on the provided segmentation masks.
- **add_observations** — This function computes the observations for each region in the segmentation masks.
- **apply** — This function applies a given function to the image data in the spatialdata object.
- **threshold** — This function applies a threshold to the image data in the spatialdata object.
- **transform_expression_matrix** — This function applies a transformation to the expression matrix in the spatialdata object.
- **filter_by_obs** — Filter the object by observations based on a given feature and filtering function.
- **grow_cells** — Grows the segmentation masks by expanding the labels in the object.
Classes:
- **PreprocessingAccessor** — The image accessor enables fast indexing and preprocessing of the spatialproteomics object.
  - get_bbox — Returns the bounds of the image container.
  - get_channels — Retrieve the specified channels from the dataset.
  - add_channel — Adds channel(s) to an existing image container.
  - add_segmentation — Adds a segmentation mask field to the xarray dataset.
  - add_layer — Adds a layer (such as a mask highlighting artifacts) to the xarray dataset.
  - add_layer_from_dataframe — Adds a dataframe as a layer to the xarray object.
  - add_observations — Adds properties derived from the segmentation mask to the image container.
  - drop_observations — No docstring.
  - add_feature — Adds a feature to the image container.
  - add_obs_from_dataframe — Adds an observation table to the image container.
  - add_quantification — Quantify channel intensities over the segmentation mask.
  - add_quantification_from_dataframe — Adds an observation table to the image container.
  - drop_layers — Drops layers from the image container.
  - threshold — Apply thresholding to the image layer of the object.
  - apply — Apply a function to each channel independently.
  - normalize — Performs a percentile normalization on each channel using the 3- and 99.
  - downsample — Downsamples the entire dataset by selecting every `rate`-th element along the x and y dimensions.
  - rescale — Rescales the image and segmentation mask in the object by a given scale.
  - filter_by_obs — Filter the object by observations based on a given feature and filtering function.
  - remove_outlying_cells — Removes outlying cells from the image container.
  - grow_cells — Grows the segmentation masks by expanding the labels in the object.
  - merge_segmentation — Merge segmentation masks.
  - get_layer_as_df — Returns the specified layer as a pandas DataFrame.
  - get_disconnected_cell — Returns the first disconnected cell from the segmentation layer.
  - transform_expression_matrix — Transforms the expression matrix based on the specified mode.
  - mask_region — Mask a region in the image.
  - mask_cells — Mask cells in the segmentation mask.
  - convert_to_8bit — Convert the image to 8-bit.

## spatialproteomics/pp/utils.py
Functions:
- **merge** — Merge multiple images into a single image using different projection methods.
- **handle_disconnected_cells** — Handle disconnected cells in a segmentation mask.

## spatialproteomics/sd/__init__.py
- (No public functions or classes found)

## spatialproteomics/sd/utils.py
- (No public functions or classes found)

## spatialproteomics/tl/__init__.py
- (No public functions or classes found)

## spatialproteomics/tl/tool.py
Functions:
- **cellpose** — This function runs the cellpose segmentation algorithm on the provided image data.
- **stardist** — This function runs the stardist segmentation algorithm on the provided image data.
- **mesmer** — This function runs the mesmer segmentation algorithm on the provided image data.
- **astir** — This function applies the ASTIR algorithm to predict cell types based on the expression matrix.
Classes:
- **ToolAccessor** — The tool accessor enables the application of external tools such as StarDist or Astir.
  - cellpose — Segment cells using Cellpose.
  - stardist — Apply StarDist algorithm to perform instance segmentation on the nuclear image.
  - mesmer — Segment cells using Mesmer.
  - astir — This method predicts cell types from an expression matrix using the Astir algorithm.
  - convert_to_anndata — Convert the spatialproteomics object to an anndata.
  - convert_to_spatialdata — Convert the spatialproteomics object to a spatialdata object.

## spatialproteomics/tl/utils.py
- (No public functions or classes found)
