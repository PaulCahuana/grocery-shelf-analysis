import sys, cv2, json
from .cv_core import compute_fullness_cv, draw_gaps

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m app.cli_fullness <image_path> [mask_path]")
        sys.exit(1)

    img_path = sys.argv[1]
    mask_path = sys.argv[2] if len(sys.argv) > 2 else None

    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"Failed to read image: {img_path}")
        sys.exit(2)

    mask = None
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    fullness, gaps = compute_fullness_cv(bgr, mask)
    print(json.dumps({"fullness_cv": round(fullness, 4), "num_gaps": len(gaps), "gaps": gaps}, indent=2))

    # Optional: save a visualization next to the image
    vis = draw_gaps(bgr, gaps)
    out_path = img_path.rsplit(".", 1)[0] + "_gaps.png"
    cv2.imwrite(out_path, vis)
    print(f"Saved visualization: {out_path}")

if __name__ == "__main__":
    main()
