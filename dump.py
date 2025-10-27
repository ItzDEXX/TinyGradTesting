# dump_params.py
import argparse, json, numpy as np
np.set_printoptions(suppress=True, linewidth=200)

def to_pylist(a, round_to=6):
    # Convert numpy array to a Python list with rounded floats (nice to paste)
    return np.round(a.astype(float), round_to).tolist()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=1, help="batch")
    p.add_argument("--L", type=int, default=4, help="tokens")
    p.add_argument("--D", type=int, default=8, help="model dim")
    p.add_argument("--Hfactor", type=int, default=4, help="hidden = Hfactor*D")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--npz", type=str, default="params_dump.npz", help="where to save all arrays")
    args = p.parse_args()

    B, L, D, H = args.B, args.L, args.D, args.Hfactor*args.D
    rs = np.random.RandomState(args.seed)

    # Input example (optional)
    x = rs.randn(B, L, D).astype(np.float32)

    # Attention (single-head) params
    Wq = rs.randn(D, D).astype(np.float32)
    Wk = rs.randn(D, D).astype(np.float32)
    Wv = rs.randn(D, D).astype(np.float32)
    # (Optional final proj Wo if you later add it)
    # Wo = rs.randn(D, D).astype(np.float32)

    # MLP params
    W1 = rs.randn(D, H).astype(np.float32)
    b1 = rs.randn(H).astype(np.float32)
    W2 = rs.randn(H, D).astype(np.float32)
    b2 = rs.randn(D).astype(np.float32)

    # Save everything to NPZ (safe for large arrays)
    np.savez(args.npz, x=x, Wq=Wq, Wk=Wk, Wv=Wv, W1=W1, b1=b1, W2=W2, b2=b2)
    print(f"\nSaved arrays to: {args.npz}\n")

    # Print copy-pasteable Python syntax (rounded)
    print("# ===== COPY BELOW INTO YOUR FIXED PARAMS FILE =====")
    print(f"B, L, D = {B}, {L}, {D}")
    print(f"H = {H}\n")
    print("x_vals  = "  + json.dumps(to_pylist(x)))
    print("Wq_vals = " + json.dumps(to_pylist(Wq)))
    print("Wk_vals = " + json.dumps(to_pylist(Wk)))
    print("Wv_vals = " + json.dumps(to_pylist(Wv)))
    print("W1_vals = " + json.dumps(to_pylist(W1)))
    print("b1_vals = " + json.dumps(to_pylist(b1)))
    print("W2_vals = " + json.dumps(to_pylist(W2)))
    print("b2_vals = " + json.dumps(to_pylist(b2)))
    print("# ===== END COPY =====\n")

if __name__ == "__main__":
    main()
