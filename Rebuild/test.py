from pathlib import Path

from NewParams import NewParams

image_path = Path(r"D:\cst2py_box\Auto_py2cst_v0.71\Rebuild\test48.png")
output_dir = Path(r"D:\cst2py_box\Auto_py2cst_v0.71\Rebuild\param48_output")

params = NewParams(image_path, save_dir=output_dir)

cp = params.parameterize(save_dir=output_dir)

results = cp.results()
json_path = cp.save_json(output_dir / "curve_parameterization.json")
visual_path = cp.visualize(output_dir / "curve_parameterization.png")

for contour in results:
    print(contour["component_id"], contour["closed"])
    for seg in contour["segments"]:
        print(seg["kind"], seg["max_error"], seg)

print(f"JSON saved to: {json_path}")
print(f"Visualization saved to: {visual_path}")
