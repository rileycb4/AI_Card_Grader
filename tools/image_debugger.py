import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

class PipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Card Detection & Crop Pipeline")
        self.root.geometry("1400x900")

        # Top Control Bar
        self.ctrl_frame = tk.Frame(self.root, pady=10)
        self.ctrl_frame.pack(side="top", fill="x")
        
        ttk.Button(self.ctrl_frame, text="1. Upload & Process", command=self.upload_and_process).pack(side="left", padx=10)
        
        # Save Button (starts disabled until an image is processed)
        self.save_btn = ttk.Button(self.ctrl_frame, text="2. Save Cropped Image", command=self.save_cropped, state="disabled")
        self.save_btn.pack(side="left", padx=10)

        self.display_frame = tk.Frame(self.root)
        self.display_frame.pack(expand=True, fill="both", padx=20, pady=20)

        self.stages = ["Original", "Grayscale + Blur", "Canny Edges", "Dilation", "Outline Detection", "Cropped Card"]
        self.labels = {}
        self.current_cropped = None # Store the CV2 image here for saving
        
        for i, stage_name in enumerate(self.stages):
            frame = tk.Frame(self.display_frame, bd=1, relief="sunken")
            row, col = divmod(i, 3)
            frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            tk.Label(frame, text=stage_name, font=("Arial", 10, "bold")).pack()
            lbl = tk.Label(frame, text="Waiting...")
            lbl.pack(expand=True)
            self.labels[stage_name] = lbl

        for i in range(3): self.display_frame.columnconfigure(i, weight=1)
        for i in range(2): self.display_frame.rowconfigure(i, weight=1)

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def get_warp(self, image, pts):
        rect = self.order_points(pts.reshape(4, 2))
        (tl, tr, br, bl) = rect
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_w = max(int(width_a), int(width_b))
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_h = max(int(height_a), int(height_b))
        dst = np.array([[0, 0], [max_w-1, 0], [max_w-1, max_h-1], [0, max_h-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (max_w, max_h))

    def upload_and_process(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not path: return

        orig = cv2.imread(path)
        self.show_step(orig, "Original")

        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (35, 35), 0)
        self.show_step(blurred, "Grayscale + Blur", is_gray=True)

        edged = cv2.Canny(blurred, 10, 15)
        self.show_step(edged, "Canny Edges", is_gray=True)

        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(edged, kernel, iterations=2)
        self.show_step(dilated, "Dilation", is_gray=True)

        outline_img = orig.copy()
        contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            cv2.drawContours(outline_img, [approx], -1, (0, 255, 0), 8)
            self.show_step(outline_img, "Outline Detection")

            if len(approx) == 4:
                self.current_cropped = self.get_warp(orig, approx)
                self.show_step(self.current_cropped, "Cropped Card")
                self.save_btn.config(state="normal") # Enable saving
            else:
                self.labels["Cropped Card"].config(image="", text="Could not find 4 corners.")
                self.save_btn.config(state="disabled")
        else:
            self.save_btn.config(state="disabled")

    def save_cropped(self):
        if self.current_cropped is None:
            return

        # Default filename based on original or generic
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
            title="Save Cropped Card"
        )
        
        if file_path:
            # OpenCV writes BGR, imwrite handles the conversion/saving
            success = cv2.imwrite(file_path, self.current_cropped)
            if success:
                messagebox.showinfo("Success", f"Image saved to:\n{file_path}")
            else:
                messagebox.showerror("Error", "Failed to save the image.")

    def show_step(self, cv_img, stage_name, is_gray=False):
        if is_gray:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        pil_img = Image.fromarray(rgb_img)
        pil_img.thumbnail((400, 400), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)
        
        self.labels[stage_name].config(image=tk_img, text="")
        self.labels[stage_name].image = tk_img

if __name__ == "__main__":
    root = tk.Tk()
    app = PipelineGUI(root)
    root.mainloop()