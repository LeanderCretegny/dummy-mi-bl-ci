import cv2
from utils.compare_utils import mse, mae

def same_renders():
    r = cv2.imread("res/renders/tests/mi_simple.png")
    r2 = r
    err, v, diff = mae(r, r2)
    assert err == 0, f"MAE of a render with itself should be 0 but got {err}"
    assert v == 0, f"Standard deviation of a render with itself should be 0 but got {err}"
    cv2.imwrite("res/out/tests/mae_same_diff.png", diff)
    
    err, v, diff = mse(r, r2)
    assert err == 0, f"MSE of a render with itself should be 0 but got {err}"
    assert v == 0, f"Standard deviation of a render with itself should be 0 but got {err}"
    cv2.imwrite("res/out/tests/mse_same_diff.png", diff)

if __name__ == "__main__":
    same_renders()