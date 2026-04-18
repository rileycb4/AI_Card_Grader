import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading

# Import the logic from the previous scripts
import train_model
#import analyze_card

class CardGraderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sports Card Quality Analyzer")
        self.root.geometry("800x600")

        # Create a Tabbed Interface
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        # Tab 1: Model Training
        self.tab_train = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_train, text="Train Model")
        self.setup_training_tab()

        # Tab 2: Card Analysis
        self.tab_analyze = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_analyze, text="Analyze Card")
        self.setup_analysis_tab()

    def setup_training_tab(self):
        # Training Data Selection
        ttk.Label(self.tab_train, text="Training Data (CSV):").grid(row=0, column=0, padx=10, pady=15, sticky="w")
        self.csv_path_var = tk.StringVar()
        ttk.Entry(self.tab_train, textvariable=self.csv_path_var, width=50).grid(row=0, column=1, padx=10, pady=15)
        ttk.Button(self.tab_train, text="Browse", command=self.browse_csv).grid(row=0, column=2, padx=10, pady=15)

        # Output Model Path
        ttk.Label(self.tab_train, text="Save Model As (.pkl):").grid(row=1, column=0, padx=10, pady=15, sticky="w")
        self.model_save_var = tk.StringVar(value="card_grader_model.pkl")
        ttk.Entry(self.tab_train, textvariable=self.model_save_var, width=50).grid(row=1, column=1, padx=10, pady=15)

        # Train Button
        self.btn_train = ttk.Button(self.tab_train, text="Train Regression Model", command=self.run_training)
        self.btn_train.grid(row=2, column=1, pady=30)

        # Status Output
        self.train_status = tk.Text(self.tab_train, height=10, width=70, state='disabled')
        self.train_status.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

    def setup_analysis_tab(self):
        # Front Image Selection
        ttk.Label(self.tab_analyze, text="Front Image:").grid(row=0, column=0, padx=10, pady=15, sticky="w")
        self.front_img_var = tk.StringVar()
        ttk.Entry(self.tab_analyze, textvariable=self.front_img_var, width=50).grid(row=0, column=1, padx=10, pady=15)
        ttk.Button(self.tab_analyze, text="Browse", command=lambda: self.browse_image(self.front_img_var)).grid(row=0, column=2, padx=10, pady=15)

        # Back Image Selection
        ttk.Label(self.tab_analyze, text="Back Image:").grid(row=1, column=0, padx=10, pady=15, sticky="w")
        self.back_img_var = tk.StringVar()
        ttk.Entry(self.tab_analyze, textvariable=self.back_img_var, width=50).grid(row=1, column=1, padx=10, pady=15)
        ttk.Button(self.tab_analyze, text="Browse", command=lambda: self.browse_image(self.back_img_var)).grid(row=1, column=2, padx=10, pady=15)

        # Analyze Button
        self.btn_analyze = ttk.Button(self.tab_analyze, text="Run Analysis", command=self.run_analysis)
        self.btn_analyze.grid(row=2, column=1, pady=30)

        # Results Display (Can be mapped to a ttk.Treeview later for batch processing)
        self.results_display = tk.Text(self.tab_analyze, height=12, width=70, state='disabled')
        self.results_display.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

    # --- Helper Functions ---
    def browse_csv(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filename:
            self.csv_path_var.set(filename)

    def browse_image(self, string_var):
        filename = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if filename:
            string_var.set(filename)

    def log_status(self, text_widget, message):
        text_widget.config(state='normal')
        text_widget.insert(tk.END, message + "\n")
        text_widget.see(tk.END)
        text_widget.config(state='disabled')

    # --- Execution Logic ---
    def run_training(self):
        csv_path = self.csv_path_var.get()
        model_path = self.model_save_var.get()
        
        if not csv_path:
            messagebox.showwarning("Input Error", "Please select a training CSV file.")
            return

        self.log_status(self.train_status, f"Starting training using data from: {csv_path}...")
        self.btn_train.config(state='disabled')

        # Run in a separate thread so the GUI doesn't freeze
        def task():
            try:
                train_model.train_grading_model(csv_path, model_path)
                self.log_status(self.train_status, "Training complete! Model saved to " + model_path)
            except Exception as e:
                self.log_status(self.train_status, f"Error during training: {str(e)}")
            finally:
                self.btn_train.config(state='normal')
        
        threading.Thread(target=task).start()

    def run_analysis(self):
        front_path = self.front_img_var.get()
        back_path = self.back_img_var.get()
        model_path = self.model_save_var.get()

        if not front_path or not back_path:
            messagebox.showwarning("Input Error", "Please select both front and back images.")
            return

        self.log_status(self.results_display, "Analyzing card...")
        self.btn_analyze.config(state='disabled')

        def task():
            try:
                # analyzer = analyze_card.CardAnalyzer(model_path)
                # results = analyzer.analyze(front_path, back_path)
                
                # Mock Results for demonstration
                results = {"Raw Score": 948.5, "Final Grade": 9.0}
                
                self.log_status(self.results_display, f"Analysis Complete.\nRaw Score: {results['Raw Score']}/1000\nFinal Grade: {results['Final Grade']}")
            except Exception as e:
                self.log_status(self.results_display, f"Error during analysis: {str(e)}")
            finally:
                self.btn_analyze.config(state='normal')
        
        threading.Thread(target=task).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = CardGraderApp(root)
    root.mainloop()