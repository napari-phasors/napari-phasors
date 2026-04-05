# Masks

Masking allows you to restrict phasor analysis to specific regions of your image. napari-phasors supports several ways to create and use masks:

## Masking with shapes

You can draw shapes (e.g., polygons, rectangles, ellipses) directly on your image using napari's built-in shapes tool. To use a shapes layer as a mask:

1. Add a new shapes layer and draw your region(s) of interest.
2. In the **Phasor Plot** widget, select the shapes layer as the mask source.
3. Only pixels inside the shapes will be included in the phasor plot and analysis.

## Masking with labels

You can use a labels layer (e.g., from manual annotation or segmentation) as a mask:

1. Add or load a labels layer where each label value defines a region.
2. In the **Phasor Plot** widget, select the labels layer as the mask source.
3. You can restrict analysis to a specific label value or include all nonzero labels.

## Masking with cursor selection

When you use the cursor selection tools (circular, polar, elliptical) in the **Selection** tab, a new labels layer is created for each cursor. These labels layers can themselves be used as masks for further analysis or for other image layers.

- For example, after defining a region with a circular cursor, select the resulting labels layer as a mask in the **Phasor Plot** widget.

## Assigning masks to image layers

- You can assign a mask to a single image layer, restricting phasor analysis to that region only.
- Different image layers can have different masks assigned, allowing for independent region-of-interest analysis across multiple images.
- To do this, select the desired image layer, then choose the appropriate mask layer in the **Phasor Plot** widget.

## Example workflow

1. Load your image(s) and create or load a mask (shapes, labels, or cursor selection).
2. Select the image layer you want to analyze.
3. Assign the mask layer to that image layer in the **Phasor Plot** widget.
4. Repeat for other image layers as needed, each with its own mask if desired.

![mask](https://github.com/napari-phasors/napari-phasors-data/raw/main/gifs/mask.gif)
