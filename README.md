# image-stacker

Image stacker used to perform image registration using classic methods.

---

Requirements:
 - cv2
 - numpy
 - tqdm


---

Methods supported:
 - Akaze
 - Oriented FAST and Rotated BRIEF (ORB)
 - Enhanced Correlation Coefficient (ECC) 

---

Example run:

```
python stack.py --method=orb --directory=./images/noisy_images --scale-percent=200 --draw-matches
```