# Codebase Overview (modules and their functions/methods)

_Generated: 2025-08-25T07:43:26Z_


## __init__.py

> _No docstring provided._

_No functions or classes found._


## _shared/segmentation.py

> _No docstring provided._


### Functions

- **create_multichannel_tiff(input_dir, output_dir, output_filename)** — Create a multi-channel TIFF image by stacking individual TIFF files from a specified directory.
- **combine_channels(image_dict, channel_list, new_channel_name)** — Combine multiple channels into a single channel.
- **format_CODEX(image, channel_names, number_cycles, images_per_cycle, input_format)** — Formats image data into a dictionary based on the specified input format.

## _shared/segmentation_unused.py

> _No docstring provided._


### Functions

- **overlay_masks_on_image(image, masks, gamma)** — _No docstring provided._
- **check_segmentation(overlay, grayscale, n, tilesize)** — _No docstring provided._

## archive/tools_archive.py

> The function tl_cell_types_de performs differential enrichment analysis for various cell subsets between different neighborhoods using linear regression.


### Functions

- **tl_cell_types_de(ct_freq, all_freqs, neighborhood_num, nbs, patients, group, cells, cells1)** — _No docstring provided._
- **tl_Create_neighborhoods(df, n_num, cluster_col, X, Y, regions, sum_cols, keep_cols, ks)** — _No docstring provided._
- **tl_Chose_window_size(windows, n_num, n_neighborhoods, sum_cols, n2_name)** — _No docstring provided._
- **tl_spatial_context_stats(n_num, patient_ID_component1, patient_ID_component2, windows, total_per_thres, comb_per_thres, tissue_column, subset_list, plot_order, pal_tis, subset_list_tissue1, subset_list_tissue2)** — _No docstring provided._
- **tl_xycorr(df, sample_col, y_rows, x_columns, X_pix, Y_pix)** — _No docstring provided._
- **tl_get_distances(df, cell_list, cell_type_col)** — _No docstring provided._
- **tl_generate_voronoi_plots(df, output_path, grouping_col, tissue_col, region_col, x_col, y_col)** — Generate Voronoi plots for unique combinations of tissue and region.
- **tl_generate_masks_from_images(image_folder, mask_output, image_type, filter_size, threshold_value)** — Generate binary masks from CODEX images.
- **tl_generate_info_dataframe(df, voronoi_output, mask_output, filter_list, info_cols)** — Generate a filtered DataFrame based on specific columns and values.
- **tl_process_files(voronoi_path, mask_path, region)** — Process files based on the provided paths and region.
- **tl_process_data(df_info, output_dir_csv)** — Process data based on the information provided in the DataFrame.
- **tl_analyze_image(path, output_dir, invert, properties_list)** — Analyze an image by performing connected component analysis on patches and storing their information.
- **tl_apply_mask(image_path, mask_path, output_path)** — Apply a mask to an image and save the resulting masked image.
- **tl_generate_mask(path, output_dir, filename, filter_size, threshold_value)** — Generate a mask from a maximum projection of an input image.
- **tl_test_clustering_resolutions(adata, clustering, n_neighbors, resolutions)** — Test different resolutions for reclustering using Louvain or Leiden algorithm.
- **tl_corr_cell_ad(adata, per_categ, grouping_col, rep, sub_column, normed, sub_list2)** — Perform correlation analysis on a pandas DataFrame and plot correlation scatter plots.

## helperfunctions/__init__.py

> _No docstring provided._

_No functions or classes found._


## helperfunctions/_general.py

> _No docstring provided._


### Functions

- **hf_generate_random_colors(n, rand_seed)** — _No docstring provided._
- **hf_assign_colors(names, colors)** — _No docstring provided._
- **hf_per_only(data, grouping, replicate, sub_col, sub_list, per_cat, norm)** — _No docstring provided._
- **hf_normalize(X)** — _No docstring provided._
- **hf_cell_types_de_helper(df, ID_component1, ID_component2, neighborhood_col, group_col, group_dict, cell_type_col)** — _No docstring provided._
- **hf_get_pathcells(query_database, query_dict_list)** — Return set of cells that match query_dict path.
- **hf_get_windows(job, n_neighbors, exps, tissue_group, X, Y)** — _No docstring provided._
- **hf_index_rank(a, axis)** — returns the index of every index in the sorted array
- **hf_znormalize(raw_cells, grouper, markers, clip, dropinf)** — _No docstring provided._
- **hf_fast_divisive_cluster(X, num_clusters, metric, prints)** — _No docstring provided._
- **hf_alloc_cells(X, centroids, metric)** — _No docstring provided._
- **hf_get_sum_cols(cell_cuts, panel)** — _No docstring provided._
- **hf_get_thresh_simps(x, thresh)** — _No docstring provided._
- **hf_prepare_neighborhood_df(cells_df, patient_ID_component1, patient_ID_component2, neighborhood_column)** — _No docstring provided._
- **hf_prepare_neighborhood_df2(cells_df, patient_ID_component1, patient_ID_component2, neighborhood_column)** — _No docstring provided._
- **hf_cor_subset(cor_mat, threshold, cell_type)** — _No docstring provided._
- **hf_get_redundant_pairs(df)** — Get diagonal and lower triangular pairs of correlation matrix
- **hf_simp_rep(data, patient_col, tissue_column, subset_list_tissue, ttl_per_thres, comb_per_thres, thres_num)** — _No docstring provided._
- **hf_get_top_abs_correlations(df, thresh)** — _No docstring provided._
- **make_anndata(df_nn, col_sum, nonFuncAb_list)** — Convert a denoised DataFrame into anndata format.
- **hf_split_channels(input_path, output_folder, channel_names_file)** — Split channels of a TIFF image and save them as separate files.
- **hf_voronoi_finite_polygons_2d(vor, radius)** — Reconstruct infinite voronoi regions in a 2D diagram to finite
- **hf_list_folders(directory)** — Retrieve a list of folders in a given directory.
- **hf_process_dataframe(df)** — Extract information from a pandas DataFrame containing file paths and IDs.
- **hf_get_png_files(directory)** — Get a list of PNG files in a given directory.
- **hf_find_tiff_file(directory, prefix)** — Find a TIFF file in a given directory with a specified prefix.
- **hf_extract_filename(filepath)** — Extract the filename from a given filepath.
- **hf_get_tif_filepaths(directory)** — Recursively searches the specified directory and its subdirectories for TIFF files (.tif) and returns a list of their file paths.
- **hf_prepare_cca(df, neighborhood_column, subsets)** — _No docstring provided._
- **invert_dictionary(dictionary)** — _No docstring provided._
- **hf_replace_names(color_dict, name_dict)** — _No docstring provided._
- **hf_annotate_cor_plot(x, y, **kws)** — _No docstring provided._
- **is_dark(color)** — Determines if a color is dark based on its RGB values.
- **check_for_gpu(tensorflow, torch)** — Check if a GPU is available for use by TensorFlow and PyTorch.

### Classes

- **Neighborhoods** — _No docstring provided._
  - **add_dummies(self)** — _No docstring provided._
  - **get_tissue_chunks(self)** — _No docstring provided._
  - **make_windows(self, job)** — _No docstring provided._
  - **k_windows(self)** — _No docstring provided._

## helperfunctions/_qptiff_converter.py

> _No docstring provided._


### Functions

- **downscale_tissue(file_path, DNAslice, downscale_factor, sigma, padding, savefig, showfig, output_dir, output_fname, figsize)** — _No docstring provided._

## plotting/__init__.py

> _No docstring provided._

_No functions or classes found._


## plotting/_general.py

> _No docstring provided._


### Functions

- **pl_stacked_bar_plot(data, per_cat, grouping, cell_list, output_dir, norm, save_name, col_order, sub_col, name_cat, fig_sizing, plot_order, color_dic, remove_leg)** — Plot a stacked bar plot based on the given data.
- **pl_swarm_box(data, grouping, per_cat, replicate, sub_col, sub_list, output_dir, norm, figure_sizing, save_name, plot_order, col_in, color_dic, flip)** — _No docstring provided._
- **pl_Shan_div(tt, test_results, res, grouping, color_dic, sub_list, output_dir, save, plot_order, fig_size)** — Plot Shannon Diversity using boxplot and swarmplot.
- **pl_cell_type_composition_vis(data, sample_column, cell_type_column, figsize, output_dir)** — Visualize cell type composition using stacked and unstacked bar plots.
- **pl_regions_per_sample(data, sample_col, region_col, bar_color)** — _No docstring provided._
- **pl_neighborhood_analysis_2(data, k_centroids, values, sum_cols, X, Y, reg, output_dir, k, plot_specific_neighborhoods, size, axis, ticks_fontsize, show_spatial_plots, palette)** — Perform neighborhood analysis and visualize results.
- **pl_highlighted_dot(df, x_col, y_col, group_col, highlight_group, highlight_color, region_col, subset_col, subset_list)** — Plots an XY dot plot colored by a grouping column for each unique region.
- **pl_create_pie_charts(data, group_column, count_column, plot_order, show_percentages, color_dict)** — Create pie charts for each group based on a grouping column, showing the percentage of total rows based on a
- **pl_cell_types_de(data, pvals, neigh_num, output_dir, figsize)** — Plot cell types differential expression as a heatmap.
- **pl_community_analysis_2(data, values, sum_cols, output_dir, k_centroids, X, Y, reg, save_path, k, size, axis, ticks_fontsize, plot_specific_community, show_spatial_plots, palette)** — Plot community analysis.
- **pl_Visulize_CCA_results(CCA_results, output_dir, save_fig, p_thresh, save_name, colors)** — Visualize the results of Canonical Correlation Analysis (CCA) using a graph.
- **pl_plot_modules_heatmap(data, cns, cts, figsize, num_tissue_modules, num_cn_modules)** — Plot the modules and their loadings using heatmaps.
- **pl_plot_modules_graphical(data, cts, cns, num_tissue_modules, num_cn_modules, scale, color_dic, save_name, save_path)** — Generate a graphical representation of modules discovered in a dataset using non-negative matrix factorization (NMF).
- **pl_evaluate_ranks(data, num_tissue_modules)** — Evaluate the reconstruction error of different ranks in non-negative matrix factorization (NMF).
- **pl_corr_cell(data, per_categ, group2, rep, sub_column, cell, output_dir, save_name, thres, normed, cell2, sub_list2)** — Perform correlation analysis on a pandas DataFrame.
- **pl_cor_plot(data, group1, per_cat, sub_col, sub_list, norm, group2, count, plot_scatter)** — Create a correlation plot using a pandas DataFrame.
- **pl_cor_subplot(mp, sub_list, output_dir, save_name)** — Create a subplot of pairwise correlation plots using a subset of columns from a pandas DataFrame.
- **annotate(data, names, **kws)** — _No docstring provided._
- **pl_cor_subplot_new(mp, sub_list, output_dir, save_name)** — Create a subplot of pairwise correlation plots using a subset of columns from a pandas DataFrame.
- **pl_Niche_heatmap(k_centroids, w, n_num, sum_cols)** — Create a heatmap to show the types of cells (ClusterIDs) in different niches.
- **pl_Barycentric_coordinate_projection(w, plot_list, threshold, output_dir, save_name, col_dic, l, n_num, cluster_col, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, figsize)** — Create a barycentric coordinate projection plot.
- **pl_get_network(ttl_per_thres, comb_per_thres, color_dic, windows, n_num, l, tissue_col, tissue_subset_list, sub_col, neigh_sub, save_name, save_path, figsize)** — Generate a network plot based on combination frequencies.
- **pl_spatial_context_stats_vis(neigh_comb, simp_df_tissue1, simp_df_tissue2, pal_tis, plot_order, figsize)** — _No docstring provided._
- **pl_conplot(df, feature, exp, X, Y, invert_y, cmap, size, alpha, figsize, exps, fig, **kwargs)** — Plot continuous variable with a colormap:
- **pl_catplot(df, hue, exp, X, Y, invert_y, size, legend, palette, figsize, style, exps, axis, ticks_fontsize, scatter_kws, **kwargs)** — Plots cells in tissue section color coded by either cell type or node allocation.
- **pl_comb_num_freq(data_list, plot_order, pal_tis, figsize)** — _No docstring provided._
- **zcount_thres(dfz, col_num, cut_off, count_bin, zsum_bin, figsize)** — Determines the threshold to use for removing noises. The default cut off is the top 1%.
- **pl_mono_cluster_spatial(df, sample_col, cluster_col, x, y, color_dict, s, alpha, figsize)** — _No docstring provided._
- **pl_visualize_2D_density_plot(df, region_column, selected_region, subsetting_column, values_list, x_column, y_column)** — _No docstring provided._
- **pl_create_cluster_celltype_heatmap(dataframe, cluster_column, celltype_column)** — _No docstring provided._
- **catplot(adata, color, unique_region, subset, X, Y, invert_y, size, alpha, palette, savefig, output_dir, output_fname, figsize, style, axis, scatter_kws, n_columns, legend_padding, rand_seed)** — Plots cells in tissue section color coded by either cell type or node allocation.
- **pl_generate_CN_comb_map(graph, tops, e0, e1, simp_freqs, palette, figsize, savefig, output_dir)** — _No docstring provided._
- **stacked_bar_plot(adata, color, grouping, cell_list, output_dir, norm, savefig, output_fname, col_order, sub_col, name_cat, fig_sizing, plot_order, palette, remove_leg, rand_seed)** — Plot a stacked bar plot based on the given data.
- **pl_swarm_box_ad(adata, grouping, per_cat, replicate, sub_col, sub_list, output_dir, norm, figure_sizing, save_name, plot_order, col_in, color_dic, flip)** — _No docstring provided._
- **create_pie_charts(adata, color, grouping, plot_order, show_percentages, palette, savefig, output_fname, output_dir, rand_seed)** — Create pie charts for each group based on a grouping column, showing the percentage of total rows based on a
- **cn_exp_heatmap(adata, cluster_col, cn_col, palette, savefig, output_fname, output_dir, row_clus, col_clus, rand_seed, figsize)** — Create a heatmap of expression data, clustered by rows and columns.
- **pl_area_nuc_cutoff(df, cutoff_area, cutoff_nuc, cellsize_column, nuc_marker_column, color_by, palette, alpha, size, log_scale)** — _No docstring provided._
- **pl_plot_scatter_correlation(data, x, y, xlabel, ylabel, save_path)** — _No docstring provided._
- **pl_plot_scatter_correlation_ad(adata, x, y, xlabel, ylabel, save_path)** — _No docstring provided._
- **pl_plot_correlation_matrix(cmat)** — _No docstring provided._
- **dumbbell(data, figsize, colors, savefig, output_fname, output_dir)** — Create a dumbbell plot.
- **plot_top_n_distances(dist_table_filt, dist_data_filt, n, colors, dodge, savefig, output_fname, output_dir, figsize, unit, errorbars)** — _No docstring provided._
- **cn_map(adata, cnmap_dict, cn_col, palette, figsize, savefig, output_fname, output_dir, rand_seed)** — Generates a CNMap plot using the provided data and parameters.
- **coordinates_on_image(df, overlay_data, color, x, y, fig_width, fig_height, dot_size, convert_to_grey, scale, cmap, savefig, output_dir, output_fname)** — Plot coordinates on an image.
- **count_patch_proximity_res(adata, x, hue, palette, order, key_name, savefig, output_dir, output_fname)** — Create a count plot for patch proximity results.
- **BC_projection(adata, cnmap_dict, cn_col, plot_list, cn_col_annt, palette, figsize, rand_seed, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, n_num, threshold, savefig, output_fname, output_dir, dpi)** — Plot barycentric projection.
- **distance_graph(dist_table, distance_pvals, palette, condition_pair, interaction_col, condition_col, logfold_group_col, celltype1_col, celltype2_col, pair_col, with_labels, node_size, font_size, multiplication_factor, savefig, output_fname, output_dir, dpi, color_seed)** — Generates a distance graph from a dataframe.
- **plot_masks(adata, color, unique_region, subset, X, Y, invert_y, palette, savefig, output_dir, output_fname, figsize, background, n_columns, rand_seed, outline)** — Plot segmentation masks colored by categorical annotation.
- **create_mask_dict(mask_file_paths, region_names)** — Create a dictionary mapping region names to their corresponding masks.
- **create_mask_dict_png(mask_file_paths, region_names)** — Create a dictionary mapping region names to their corresponding masks.
- **ppa_res_donut(adata, cat_col, key_name, palette, distance_mode, unit, figsize, add_guides, text, label_color, rand_seed, subset_column, subset_condition, group_by, title, savefig, output_fname, output_dir)** — Plot donut chart of cell type proportions at different distances.

## plotting/_qptiff_converter.py

> _No docstring provided._


### Functions

- **tissue_lables(tissueframe, region)** — Plot the tissue and region labels of the given DataFrame.

## plotting/_segmentation.py

> _No docstring provided._


### Functions

- **segmentation_ch(file_name, channel_file, output_dir, savefig, output_fname, extra_seg_ch_list, nuclei_channel, input_format)** — Plot the channel selected for segmentation.
- **show_masks(seg_output, nucleus_channel, additional_channels, show_subsample, n, tilesize, idx, rand_seed)** — Visualize the segmentation results of an image.

## preprocessing/__init__.py

> _No docstring provided._

_No functions or classes found._


## preprocessing/_general.py

> _No docstring provided._


### Functions

- **read_segdf(segfile_list, seg_method, region_list, meta_list)** — Read the data frame output from the segmentation functions.
- **filter_data(df, nuc_thres, size_thres, nuc_marker, cell_size, region_column, color_by, palette, alpha, size, log_scale, plot)** — Filter data based on nuclear threshold and size threshold, and visualize the data before and after filtering.
- **format(data, list_out, list_keep, method, ArcSin_cofactor)** — This function formats the data based on the specified method. It supports four methods: "zscore", "double_zscore", "MinMax", and "ArcSin".
- **xycorr(data, y_rows, x_columns, X_pix, Y_pix)** — Corrects the x and y coordinates of the data for "classic CODEX" where samples are covered by multiple regions.
- **remove_noise(df, col_num, z_sum_thres, z_count_thres)** — Removes noisy cells from the dataset based on the given thresholds.
- **compensate_cell_matrix(df, image_dict, masks, overwrite, device)** — Compensate cell matrix by computing channel means and sums.

### Classes

- **ImageProcessor** — A class used to process images and compute channel means and sums.
  - **update_adjacency_value(self, adjacency_matrix, original, neighbor)** — Updates the adjacency matrix based on the original and neighbor values.
  - **update_adjacency_matrix(self, plane_mask_flattened, width, height, adjacency_matrix, index)** — Updates the adjacency matrix based on the flattened plane mask.
  - **compute_channel_means_sums_compensated(self, image, device)** — Computes and compensates channel means and sums for each cell in a multi-channel image using

## tools/__init__.py

> _No docstring provided._

_No functions or classes found._


## tools/_general.py

> _No docstring provided._


### Functions

- **tl_calculate_neigh_combs(w, l, n_num, threshold, per_keep_thres)** — Calculate neighborhood combinations based on a threshold.
- **tl_build_graph_CN_comb_map(simp_freqs, thresh_freq)** — Build a directed graph for the CN combination map.
- **clustering(adata, clustering, marker_list, resolution, n_neighbors, reclustering, key_added, key_filter, subset_cluster, seed, fs_xdim, fs_ydim, fs_rlen, **cluster_kwargs)** — Perform clustering on the given annotated data matrix.
- **neighborhood_analysis(adata, unique_region, cluster_col, X, Y, k, n_neighborhoods, elbow, metric)** — Compute for Cellular neighborhoods (CNs).
- **build_cn_map(adata, cn_col, unique_region, palette, k, X, Y, threshold, per_keep_thres, sub_list, sub_col, rand_seed)** — Generate a cellular neighborhood (CN) map.
- **tl_format_for_squidpy(adata, x_col, y_col)** — Format an AnnData object for use with Squidpy.
- **compute_triangulation_edges(df_input, x_pos, y_pos)** — Compute unique Delaunay triangulation edges from input coordinates.
- **annotate_triangulation_vectorized(edges_df, df_input, id_col, x_pos, y_pos, cell_type_col, region)** — Annotate edges with cell metadata in a vectorized manner.
- **calculate_triangulation_distances(df_input, id, x_pos, y_pos, cell_type, region)** — Calculate and annotate triangulation distances for cells.
- **process_region(df, unique_region, id, x_pos, y_pos, cell_type, region)** — Process triangulation distances for a specific region.
- **get_triangulation_distances(df_input, id, x_pos, y_pos, cell_type, region, num_cores, correct_dtype)** — Compute triangulation distances for each unique region with parallel processing.
- **shuffle_annotations(df_input, cell_type, region, permutation)** — Shuffle cell type annotations within each region.
- **tl_iterate_tri_distances(df_input, id, x_pos, y_pos, cell_type, region, num_cores, num_iterations)** — Perform iterative permutation analysis for triangulation distances.
- **add_missing_columns(triangulation_distances, metadata, shared_column)** — Add missing metadata columns to the triangulation distances DataFrame.
- **calculate_pvalue(row)** — Calculate the p-value using the Mann-Whitney U test.
- **identify_interactions(adata, cellid, x_pos, y_pos, cell_type, region, comparison, min_observed, distance_threshold, num_cores, num_iterations, key_name, correct_dtype, aggregate_per_cell)** — Identify significant cell-cell interactions based on spatial distances.
- **adata_cell_percentages(adata, column_percentage)** — Calculate the percentage of each cell type in an AnnData object.
- **filter_interactions(distance_pvals, pvalue, logfold_group_abs, comparison)** — Filters interactions based on p-value, logfold change, and other conditions.
- **remove_rare_cell_types(adata, distance_pvals, cell_type_column, min_cell_type_percentage)** — Remove cell types with a percentage lower than the specified threshold from the distance_pvals DataFrame.
- **stellar_get_edge_index(pos, distance_thres, max_memory_usage, chunk_size)** — Constructs edge indexes in one region based on pairwise distances and a distance threshold.
- **adata_stellar(adata_train, adata_unannotated, celltype_col, region_column, x_col, y_col, sample_rate, distance_thres, epochs, num_seed_class, key_added, STELLAR_path, max_memory_usage, chunk_size, wd, lr, seed, batch_size)** — Apply the STELLAR algorithm to annotated and unannotated spatial single-cell data.
- **ml_train(adata_train, label, test_size, random_state, nan_policy_y, mode, showfig, figsize, n_neighbors)** — Train a classifier (SVC, LinearSVC, or KNN) on the data.
- **ml_predict(adata_val, svc, save_name, return_prob_mat)** — Predict labels for a given dataset using a trained Support Vector Classifier (SVC) model.
- **masks_to_outlines_scikit_image(masks)** — get outlines of masks as a 0-1 array
- **download_file_tm(url, save_path)** — Download a file from a given URL and save it to a specified path.
- **check_download_tm_plugins()** — Check and download the TissUUmaps plugins if they are not already present.
- **tm_viewer(adata, images_pickle_path, directory, region_column, region, xSelector, ySelector, color_by, keep_list, include_masks, open_viewer, add_UMAP, use_jpg_compression)** — Prepare and visualize spatial transcriptomics data using TissUUmaps.
- **tm_viewer_catplot(adata, directory, region_column, x, y, color_by, open_viewer, add_UMAP, keep_list)** — Generate and visualize categorical plots using TissUUmaps.
- **install_gpu_leiden(CUDA)** — Install the necessary packages for GPU-accelerated Leiden clustering.
- **anndata_to_GPU(adata, layer, convert_all)** — Transfers matrices and arrays to the GPU
- **anndata_to_CPU(adata, layer, convert_all, copy)** — Transfers matrices and arrays from the GPU
- **install_stellar(CUDA)** — _No docstring provided._
- **launch_interactive_clustering(adata, output_dir)** — Launch an interactive clustering application for single-cell data analysis.
- **apply_dbscan_clustering(df, min_samples, x_col, y_col, allow_single_cluster)** — Apply DBSCAN clustering to a dataframe and update the cluster labels in the original dataframe.
- **identify_points_in_proximity(df, full_df, identification_column, cluster_column, x_column, y_column, radius, edge_neighbours, plot, concave_hull_length_threshold, concavity)** — Identify points in proximity within clusters and generate result and outline DataFrames.
- **precompute(df, x_column, y_column, full_df, identification_column, edge_neighbours)** — Precompute nearest neighbors and unique clusters.
- **process_cluster(args, nbrs, unique_clusters)** — _No docstring provided._
- **identify_hull_points(df, cluster_column, x_col, y_col, concave_hull_length_threshold, concavity)** — Identify hull points with improved performance.
- **convert_dataframe_to_geojson(df, output_dir, region_name, x, y, sample_col, region_col, patch_col, geojson_prefix, save_geojson)** — Convert a DataFrame into GeoJSON format with optional saving to file.
- **process_geojson_region(region_df, region, region_col, patch_col, x, y, sample_label)** — Process a single region to generate GeoJSON features.
- **extract_region_number(unique_region_value)** — Extract the numeric part of a region identifier.
- **analyze_peripheral_cells(patches_gdf, codex_gdf, buffer_distances, original_unit_scale, tolerance_distance)** — Analyze peripheral cells with parallel processing.
- **save_peripheral_cells(results, unit_name, region_name, output_dir, save_csv)** — Save peripheral cells for each buffer distance to CSV files.
- **process_region_peripheral_cells(args)** — Process peripheral cells for a given region (for parallel processing).
- **extract_unit_name(geojson)** — Extract a unit name from a GeoJSON object.
- **patch_proximity_analysis(adata, region_column, patch_column, group, min_cluster_size, x_column, y_column, radius, edge_neighbours, plot, savefig, output_dir, output_fname, save_geojson, allow_single_cluster, method, concave_hull_length_threshold, concavity, original_unit_scale, tolerance_distance, key_name)** — Performs a proximity analysis on patches of a given group within each region of a dataset.
- **create_visualization_hull_expansion(region, group, df_community, hull, patches_gdf, df_region, buffer_geometries, peripheral_results, x_column, y_column, buffer_distances, original_unit_scale, figsize)** — Create comprehensive visualization of the patch proximity analysis.
- **create_visualization_border_cell_radius(region_name, group_name, df_community, df_full, cluster_column, identification_column, x_column, y_column, radius, hull_points, proximity_results, hull_neighbors, figsize)** — Create a multi-panel visualization for the border cell radius proximity analysis method.

## tools/_qptiff_converter.py

> _No docstring provided._


### Functions

- **label_tissue(resized_im, lower_cutoff, upper_cutoff, savefig, showfig, output_dir, output_fname)** — Label the tissue in the given image.
- **save_labelled_tissue(filepath, tissueframe, region, padding, downscale_factor, output_dir, output_fname)** — Save the labelled tissue from the given image.

## tools/_segmentation.py

> _No docstring provided._


### Functions

- **load_image_dictionary(file_name, channel_file, input_format, nuclei_channel)** — Loads images and channel names based on the specified format using tifffile.
- **setup_gpu(use_gpu, set_memory_growth)** — Configures TensorFlow GPU memory growth to avoid allocating all memory at once.
- **prepare_segmentation_dict(image_dict, nuclei_channel, membrane_channel_list)** — Prepares a dictionary containing only the channels needed for segmentation.
- **resize_segmentation_images(seg_dict, resize_factor)** — Resizes images within the segmentation dictionary using area interpolation.
- **resize_mask(mask, target_shape_or_ref_img)** — Resizes a segmentation mask to a target shape using nearest neighbor interpolation.
- **generate_tiles(image_shape, tile_size, tile_overlap)** — Generate tile coordinates (y_start, y_end, x_start, x_end) with overlap.
- **display_tile_progress(tiles_info, completed_tiles_indices, image_shape, current_tile_index)** — Display an ASCII grid showing tile processing progress in Jupyter/IPython.
- **cellpose_segmentation(image_dict, output_dir, membrane_channel_name, cytoplasm_channel_name, nucleus_channel_name, use_gpu, model, custom_model, diameter, save_mask_as_png)** — Perform cell segmentation using Cellpose. Handles channel selection for Cellpose input.
- **run_cellpose(image, output_dir, use_gpu, model, custom_model, diameter, channels, save_mask_as_png)** — Internal helper to initialize and run the Cellpose model evaluation.
- **load_mesmer_model(model_dir)** — Loads the Mesmer model from a specified directory. Downloads if not found.
- **mesmer_segmentation(nuclei_image, membrane_image, image_mpp, plot_predictions, compartment, model_path)** — Perform segmentation using the DeepCell Mesmer model.
- **stitch_masks(tiles_info, tile_masks, full_shape, tile_overlap, sigma)** — Stitch multiple segmentation masks from overlapping tiles with confidence-based blending.
- **remove_border_objects(mask)** — Remove labeled objects in a mask that directly touch its borders.
- **extract_features(image_dict, segmentation_masks, channels_to_quantify, output_file, size_cutoff, use_tiling_for_intensity, tile_size, tile_overlap, memory_limit_gb)** — Extract morphological and intensity features from segmented images with memory optimization.
- **cell_segmentation(file_name, channel_file, output_dir, output_fname, seg_method, nuclei_channel, input_format, membrane_channel_list, cytoplasm_channel_list, size_cutoff, compartment, plot_predictions, model, use_gpu, diameter, save_mask_as_png, model_path, resize_factor, custom_model, differentiate_nucleus_cytoplasm, tile_size, tile_overlap, tiling_threshold, image_mpp, stitch_sigma, remove_tile_border_objects, feature_tile_size, feature_tile_overlap, feature_memory_limit_gb, set_memory_growth)** — Perform cell segmentation using Mesmer or Cellpose with optional tiling and feature extraction.