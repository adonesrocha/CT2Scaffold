CT2Scaffold is a Python pipeline designed to transform microCT (micro-computed tomography) images into customizable 3D scaffold models for 3D printing application.

The code was developed as part of a joint PhD research project, focusing on trabecular geometry.

Features:

- Load binarized microCT data (3D matrices).
- Apply ROI selection (including circular masks (in development)).
- Generate isosurfaces using `marching_cubes` algorithm.
- Modify trabecular morphology via reaction-diffusion equations.
- Close open surfaces by filling volume boundaries.
- Visualize in 3D using `matplotlib`.
- Export 3D models to `.stl` format.

Libraries Used:

- Python 3.10+
- NumPy
- SciPy
- Matplotlib
- Scikit-image
- Trimesh

How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/CT2Scaffold.git
   cd CT2Scaffold
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Edit the main script (`ct2scaffold.py`) to insert the path to your microCT data. Adjust the format of the image in the code.

4. Run the script:
   ```bash
   python ct2scaffold.py
   ```

5. Exported `.stl` files will be available in the `outputs/` folder.

Example:

You can use your own binarizedmicroCT images or adapt the code to include binarization from original images. A trabecular file as example is provided. The trabecular modification is applied automatically after loading the 3D matrix. The parameters in the Gray-Scott model were selected to improve the connectivity and trabecular thickness while maintain the trabecular architecture. The stl file exported should be more printable than the original file using the Fused Filament Fabrication 3D printing technique.

The time for processing can exceed 30 minutes depending of the file.




Author

Adones Rocha  
PhD student in Engineering / Biomaterials  
Universidade Federal do Rio de Janeiro and INSA Toulouse (joint PhD)  
adones.rocha@coppe.ufrj.br and almeida-roch@insa-toulouse.fr