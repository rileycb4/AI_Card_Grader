import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

class ProCardGrader:
    def __init__(self, root):
        self.root = root
        self.root.title("Precision Card Grading System - 1000pt Scale")
        self.root.geometry("1400x950")
        
        self.paths = {"Front": None, "Back": None}
        self.scores = {"Front": {}, "Back": {}}
        self.details = {"Front": {}, "Back": {}}
        self.raw_imgs = {"Front": None, "Back": None}
        self.analyzed_imgs = {"Front": None, "Back": None}
        self.show_overlays = {"Front": tk.BooleanVar(value=True), "Back": tk.BooleanVar(value=True)}

        self.notebook = ttk.Notebook(self.root)
        self.tab_score = ttk.Frame(self.notebook)
        self.tab_front = ttk.Frame(self.notebook)
        self.tab_back = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_score, text="1. Grading Report")
        self.notebook.add(self.tab_front, text="2. Front Analysis")
        self.notebook.add(self.tab_back, text="3. Back Analysis")
        self.notebook.pack(expand=True, fill="both")

        self.setup_score_tab()
        self.front_lbl, self.front_data = self.setup_analysis_pane(self.tab_front, "Front")
        self.back_lbl, self.back_data = self.setup_analysis_pane(self.tab_back, "Back")

    def setup_score_tab(self):
        ctrl_frame = ttk.Frame(self.tab_score, padding=20)
        ctrl_frame.pack(fill="x")

        # Front Upload with Filename Label
        f_frame = ttk.Frame(ctrl_frame)
        f_frame.pack(fill="x", pady=5)
        ttk.Button(f_frame, text="Upload Front Image", command=lambda: self.load_side("Front")).pack(side="left")
        self.f_name_lbl = ttk.Label(f_frame, text=" No file selected", foreground="gray")
        self.f_name_lbl.pack(side="left", padx=10)

        # Back Upload with Filename Label
        b_frame = ttk.Frame(ctrl_frame)
        b_frame.pack(fill="x", pady=5)
        ttk.Button(b_frame, text="Upload Back Image", command=lambda: self.load_side("Back")).pack(side="left")
        self.b_name_lbl = ttk.Label(b_frame, text=" No file selected", foreground="gray")
        self.b_name_lbl.pack(side="left", padx=10)
        
        # Global Controls
        g_frame = ttk.Frame(ctrl_frame)
        g_frame.pack(fill="x", pady=15)
        self.btn_grade = ttk.Button(g_frame, text="RUN FULL AUDIT", command=self.run_audit, state="disabled")
        self.btn_grade.pack(side="left")

        self.report_box = tk.Text(self.tab_score, font=("Consolas", 12), wrap="word", bg="#f8f9fa", padx=15, pady=15)
        self.report_box.pack(expand=True, fill="both", padx=20, pady=20)

    def setup_analysis_pane(self, parent_tab, side):
        top_frame = ttk.Frame(parent_tab, padding=8)
        top_frame.pack(fill="x")
        ttk.Checkbutton(top_frame, text="Show Analysis Overlay", variable=self.show_overlays[side],
                        command=self.refresh_visuals).pack(side="left")

        pane = tk.PanedWindow(parent_tab, orient=tk.HORIZONTAL, sashwidth=8, bg="#cccccc")
        pane.pack(expand=True, fill="both", padx=10, pady=10)
        
        img_frame = tk.Frame(pane, bg="#121212") 
        img_lbl = tk.Label(img_frame, bg="#121212")
        img_lbl.pack(expand=True)
        pane.add(img_frame, minsize=850)
        
        data_frame = tk.Frame(pane)
        data_txt = tk.Text(data_frame, font=("Courier", 11), bg="#1e1e1e", fg="#00ff00", padx=15, pady=15, width=45)
        data_txt.pack(expand=True, fill="both")
        pane.add(data_frame, minsize=350)
        
        return img_lbl, data_txt

    def load_side(self, side):
        path = filedialog.askopenfilename(title=f"Select {side} Image", filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if not path: return
        
        self.paths[side] = path
        filename = os.path.basename(path)
        
        lbl_target = self.f_name_lbl if side == "Front" else self.b_name_lbl
        lbl_target.config(text=f" {filename}", foreground="black")
            
        img = cv2.imread(path)
        self.raw_imgs[side] = img
        self.analyzed_imgs[side] = None
        self.refresh_visuals()
        
        if self.paths["Front"] and self.paths["Back"]:
            self.btn_grade.config(state="normal")

    def refresh_visuals(self):
        for side in ["Front", "Back"]:
            lbl = self.front_lbl if side == "Front" else self.back_lbl
            overlay_enabled = self.show_overlays[side].get()
            img_to_show = self.analyzed_imgs[side] if (overlay_enabled and self.analyzed_imgs[side] is not None) else self.raw_imgs[side]
            if img_to_show is not None:
                self.update_image(img_to_show, lbl)

    def run_audit(self):
        for side in ["Front", "Back"]:
            display_img, metrics, details = self.analyze_engine(self.paths[side])
            self.scores[side] = metrics
            self.details[side] = details
            self.analyzed_imgs[side] = display_img
            
            txt_panel = self.front_data if side == "Front" else self.back_data
            self.populate_context_panel(txt_panel, side, metrics, details)

        self.refresh_visuals()
        self.write_final_report()
        messagebox.showinfo("Audit Complete", "Analysis finished. Check tabs for details.")

    def analyze_engine(self, img_path):
        img = cv2.imread(img_path)
        display_img = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        details = {}

        # ------------------------
        # Corner evaluation context
        # ------------------------
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        dst = cv2.cornerHarris(blurred, 2, 3, 0.04)
        dst_norm = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=20, qualityLevel=0.01, minDistance=15,
                                          blockSize=3, useHarrisDetector=True, k=0.04)
        expected = np.array([[10, 10], [w - 11, 10], [10, h - 11], [w - 11, h - 11]], dtype=np.float32)
        off_corners = 0
        corner_distances = []
        corner_strengths = []

        if corners is not None:
            corners = np.int0(corners)
            for c in corners:
                x, y = c.ravel()
                distance = float(np.min(np.linalg.norm(expected - np.array([x, y]), axis=1)))
                corner_distances.append(distance)
                strength = float(dst[y, x]) if 0 <= y < h and 0 <= x < w else 0.0
                corner_strengths.append(strength)
                color = (0, 255, 0) if distance < 45 else (0, 0, 255)
                if distance >= 45:
                    off_corners += 1
                radius = int(np.clip(strength * 12000, 3, 12))
                cv2.circle(display_img, (x, y), radius, color, 2)
                cv2.putText(display_img, f"{strength:.0f}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        avg_corner_dist = np.mean(corner_distances) if corner_distances else 0.0
        avg_corner_strength = np.mean(corner_strengths) if corner_strengths else 0.0
        corner_score = np.clip(np.mean(dst) * 10000000, 700, 1000)
        details.update({
            "Corner Count": len(corner_distances),
            "Off Corners": off_corners,
            "Avg Corner Dist": f"{avg_corner_dist:.1f}",
            "Corner Strength": f"{avg_corner_strength:.1f}" if avg_corner_strength else "N/A"
        })

        # ------------------------
        # Edge detection and concern evaluation
        # ------------------------
        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=int(min(w, h) * 0.35), maxLineGap=25)
        edge_score = 700
        edge_angles = []
        line_count = 0

        if lines is not None:
            line_count = len(lines)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.degrees(np.arctan2((y2 - y1), (x2 - x1))))
                norm_angle = min(angle, 180 - angle)
                edge_angles.append(norm_angle)
                cv2.line(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            avg_angle = float(np.mean(edge_angles)) if edge_angles else 0.0
            deviation = np.mean([min(abs(a - 0), abs(a - 90)) for a in edge_angles]) if edge_angles else 45.0
            edge_score = np.clip(1000 - deviation * 4 - abs(line_count - 4) * 18, 650, 1000)
            details.update({
                "Edge Line Count": line_count,
                "Edge Angle Dev": f"{deviation:.1f}°",
                "Edge Note": "Detected primary board edges and straight border lines."
            })
        else:
            details.update({
                "Edge Line Count": 0,
                "Edge Angle Dev": "N/A",
                "Edge Note": "Border lines were weak or obscured; this may indicate edge wear or scanning artifact."
            })

        # 3. ADVANCED CENTERING (Content Union Logic)
        centering_score = 950
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        card_box = None
        if len(cnts) >= 1:
            for c in cnts:
                x, y, wo, ho = cv2.boundingRect(c)
                area = cv2.contourArea(c)
                if area > (w * h) * 0.2 and wo > w * 0.7 and ho > h * 0.7:
                    card_box = (x, y, wo, ho)
                    break
            if card_box is None:
                x, y, wo, ho = cv2.boundingRect(cnts[0])
                card_box = (x, y, wo, ho)

        if card_box is not None:
            x, y, wo, ho = card_box
            card_center_x = x + wo / 2
            card_center_y = y + ho / 2

            adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 15, 8)
            content_mask = adaptive
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
            content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            content_mask = cv2.erode(content_mask, np.ones((7, 7), np.uint8), iterations=1)

            content_cnts, _ = cv2.findContours(content_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            content_cnts = sorted(content_cnts, key=cv2.contourArea, reverse=True)
            filtered = []
            min_inner = max((wo * ho) * 0.0004, 110)
            max_inner = (wo * ho) * 0.6
            border = max(12, int(min(wo, ho) * 0.012))

            for c in content_cnts:
                area = cv2.contourArea(c)
                if area < min_inner or area > max_inner:
                    continue
                cx, cy, cw, ch = cv2.boundingRect(c)
                if area > (wo * ho) * 0.45 and cx <= x + border and cy <= y + border and cx + cw >= x + wo - border and cy + ch >= y + ho - border:
                    continue
                if cx < x + 4 or cy < y + 4 or cx + cw > x + wo - 4 or cy + ch > y + ho - 4:
                    continue
                filtered.append((cx, cy, cw, ch))

            if filtered:
                ix = min(box[0] for box in filtered)
                iy = min(box[1] for box in filtered)
                ix2 = max(box[0] + box[2] for box in filtered)
                iy2 = max(box[1] + box[3] for box in filtered)
                wi, hi = ix2 - ix, iy2 - iy

                content_center_x = ix + wi / 2
                content_center_y = iy + hi / 2
                x_offset = abs(content_center_x - card_center_x)
                y_offset = abs(content_center_y - card_center_y)

                x_norm = max(0.0, 1.0 - (x_offset / (wo / 2)))
                y_norm = max(0.0, 1.0 - (y_offset / (ho / 2)))
                centering_score = np.clip(((x_norm + y_norm) / 2) * 1000, 650, 1000)

                left_pad = ix - x
                right_pad = (x + wo) - ix2
                top_pad = iy - y
                bottom_pad = (y + ho) - iy2
                l_per = left_pad / float(wo) * 100
                r_per = right_pad / float(wo) * 100
                t_per = top_pad / float(ho) * 100
                b_per = bottom_pad / float(ho) * 100
                details.update({
                    "L/R Ratio": f"{l_per:.1f}%/{r_per:.1f}%", "L/R Pixels": f"L:{left_pad} R:{right_pad}",
                    "T/B Ratio": f"{t_per:.1f}%/{b_per:.1f}%", "T/B Pixels": f"T:{top_pad} B:{bottom_pad}",
                    "Content Count": len(filtered),
                    "Content Box": f"{wi}x{hi}", "Inner Bounds": f"({ix},{iy})",
                    "Centering Note": "Union of printed content including outside art text, but not the full card background."
                })

                cv2.rectangle(display_img, (x, y), (x + wo, y + ho), (0, 255, 0), 2)
                cv2.rectangle(display_img, (ix, iy), (ix2, iy2), (255, 0, 0), 2)
                cv2.line(display_img, (int(card_center_x), y), (int(card_center_x), y + ho), (0, 165, 255), 1)
                cv2.line(display_img, (x, int(card_center_y)), (x + wo, int(card_center_y)), (0, 165, 255), 1)
            else:
                details.update({
                    "L/R Ratio": "N/A", "L/R Pixels": "N/A",
                    "T/B Ratio": "N/A", "T/B Pixels": "N/A",
                    "Content Count": 0,
                    "Content Box": "N/A", "Inner Bounds": "N/A",
                    "Centering Note": "No stable printed content extents could be extracted; possible scan noise or low contrast."
                })
                cv2.rectangle(display_img, (x, y), (x + wo, y + ho), (0, 255, 0), 2)
        else:
            details.update({
                "L/R Ratio": "N/A", "L/R Pixels": "N/A",
                "T/B Ratio": "N/A", "T/B Pixels": "N/A",
                "Content Count": 0,
                "Content Box": "N/A", "Inner Bounds": "N/A",
                "Centering Note": "No clear card boundary detected."
            })

        surf_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics = {"Corners": corner_score, "Edges": edge_score, "Centering": centering_score, "Surface": np.clip(1000 - (surf_var / 2), 650, 1000)}

        # ------------------------
        # Surface defect context
        # ------------------------
        lap = cv2.Laplacian(blurred, cv2.CV_64F)
        surf_var = lap.var()
        if surf_var > 300:
            surface_note = "High variability: scratches, print grain, or texture noise likely present."
        elif surf_var > 230:
            surface_note = "Moderate texture variation: inspect for light scuffs or printing inconsistency."
        else:
            surface_note = "Surface appears visually consistent with low defect signal."
        details.update({
            "Surface Variance": f"{surf_var:.1f}",
            "Surface Status": surface_note
        })

        surface_score = np.clip(1000 - (surf_var / 2), 650, 1000)
        metrics = {
            "Corners": corner_score,
            "Edges": edge_score,
            "Centering": centering_score,
            "Surface": surface_score
        }
        return display_img, metrics, details

    def update_image(self, cv_img, label):
        """Robust image scaling to prevent black screen issue."""
        if cv_img is None: return
        self.root.update_idletasks() # Ensure geometry is calculated
        
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        
        # Determine available space
        w = label.master.winfo_width()
        h = label.master.winfo_height()
        if w < 50 or h < 50: w, h = 800, 600 # Fallback
        
        pil.thumbnail((w-20, h-20), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil)
        label.config(image=tk_img)
        label.image = tk_img

    def populate_context_panel(self, text_widget, side, metrics, details):
        text_widget.config(state="normal")
        text_widget.delete(1.0, tk.END)
        content = f"=== {side.upper()} CONTENT ANALYSIS ===\n\n"
        content += f"Corner count: {details.get('Corner Count', 0)}   Anomalous corners: {details.get('Off Corners', 'N/A')}\n"
        content += f"Corner strength average: {details.get('Corner Strength', 'N/A')}   Avg dist: {details.get('Avg Corner Dist', 'N/A')} px\n\n"
        content += f"Edge lines: {details.get('Edge Line Count', 'N/A')}   Angle deviation: {details.get('Edge Angle Dev', 'N/A')}\n"
        content += f"Edge note: {details.get('Edge Note', 'N/A')}\n\n"
        content += f"Surface variance: {details.get('Surface Variance', 'N/A')}\n"
        content += f"Surface note: {details.get('Surface Status', 'N/A')}\n\n"
        content += f"Detected inner elements (art + exterior text): {details.get('Content Count', 0)}\n"
        content += f"Content box: {details.get('Content Box', 'N/A')} at {details.get('Inner Bounds', 'N/A')}\n"
        content += f"L/R padding: {details.get('L/R Ratio', 'N/A')}   {details.get('L/R Pixels', '')}\n"
        content += f"T/B padding: {details.get('T/B Ratio', 'N/A')}   {details.get('T/B Pixels', '')}\n\n"
        content += f"Centering note: {details.get('Centering Note', 'N/A')}\n"
        content += "=" * 25 + "\n"
        content += f"COMPUTED SCORES:\nCentering: {metrics['Centering']:.1f}\nCorners:   {metrics['Corners']:.1f}\n"
        content += f"Edges:     {metrics['Edges']:.1f}\nSurface:   {metrics['Surface']:.1f}\n"
        text_widget.insert(tk.END, content)
        text_widget.config(state="disabled")

    def write_final_report(self):
        self.report_box.delete(1.0, tk.END)
        self.report_box.insert(tk.END, "--- FULL 1000-POINT AUDIT ---\n\n")
        f, b = self.scores["Front"], self.scores["Back"]
        f_det, b_det = self.details["Front"], self.details["Back"]

        for m in ["Corners", "Edges", "Centering", "Surface"]:
            avg = (f[m] + b[m]) / 2
            self.report_box.insert(tk.END, f"[{m.upper()}] Composite Score: {avg:.1f}/1000\n")
            if m == "Corners":
                self.report_box.insert(tk.END, f"Corner note: Front has {f_det.get('Off Corners', 0)} off-position corners, Back has {b_det.get('Off Corners', 0)}.\n")
                self.report_box.insert(tk.END, "Evaluation uses Harris strengths and spatial consistency relative to expected card corners.\n")
            elif m == "Edges":
                self.report_box.insert(tk.END, f"Edge note: Front detected {f_det.get('Edge Line Count', 'N/A')} border segments, Back detected {b_det.get('Edge Line Count', 'N/A')} segments.\n")
                self.report_box.insert(tk.END, "Checks line straightness, edge continuity and whether any border segment is weak or broken.\n")
            elif m == "Centering":
                self.report_box.insert(tk.END, "Centering note: This score is based on the union of artwork and outside text elements, not only the artwork frame.\n")
                self.report_box.insert(tk.END, "A strong centering result means the entire printed package sits symmetrically inside the card cut.\n")
            elif m == "Surface":
                self.report_box.insert(tk.END, "Surface note: Laplacian texture variance searches for scuffs, print grain, scratches, and uneven coating.\n")
            self.report_box.insert(tk.END, "-"*40 + "\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = ProCardGrader(root)
    root.mainloop()