import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import cv2
import numpy as np
import pygetwindow as gw
from mss import mss
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import gc  # Garbage collection

from tensorflow.python.keras.backend import clear_session
from sklearn.metrics.pairwise import euclidean_distances

# Adjust the images folder
IMAGES_FOLDER = Path(__file__).parent / "images"

DIM_Y = 1309
DIM_X = 707

rect_strings = [
    "bottom right:",
    "top left:    ",
    "top middle:  ",
    "top right:   ",
    "bottom left: "
]


def hybrid_similarity(feature1, feature2, top_k=5, weight_cosine=0.7, weight_euclidean=0.3):
    """
    Calculate a hybrid similarity score using cosine similarity and Euclidean distance.

    :param feature1: Array of features (1xN).
    :param feature2: Array of features (MxN) (templates).
    :param top_k: Number of top matches to average.
    :param weight_cosine: Weight for cosine similarity (0-1).
    :param weight_euclidean: Weight for Euclidean distance (0-1).
    :return: Weighted similarity score.
    """
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(feature1, feature2)[0]

    # Calculate Euclidean distance (convert to similarity by inverting and normalizing)
    euclidean_dist = euclidean_distances(feature1, feature2)[0]
    euclidean_sim = 1 / (1 + euclidean_dist)  # Normalize to [0, 1]

    # Combine both similarities with weights
    combined_sim = weight_cosine * cosine_sim + weight_euclidean * euclidean_sim

    # Average the top-k scores
    top_k_scores = sorted(combined_sim, reverse=True)[:top_k]
    return sum(top_k_scores) / len(top_k_scores)


class WindowFinderApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Window Finder")
        self.executor = None  # To manage threads
        self.running = False  # Flag to indicate if the script is running

        self.RECT_REGIONS=[
            [0.5162659123055162, 0.5106951871657754, 0.7940792079207921, 0.7141955691367456],  # 5
            [0.07047807637906649, 0.28962414056531705, 0.34016973125884015, 0.49173567608861724], #1
            [0.3642149929278642, 0.28812414056531704, 0.6386138613861386, 0.49273567608861724], #2
            [467.7 / DIM_X, (403 - 26.5) / DIM_Y, 660 / DIM_X, (672.8 - 26.5) / DIM_Y],# KEEP #3
            [0.21874964639321076, 0.5126951871657754, 0.48114851485148513, 0.7151955691367458] #4

        ]

        self.match_threshold = 0.63
        self.min_in_pack = 5

        self.textt = tk.Label(self.root, text="Ivy's God-Pack Finder - 3.4.1")
        self.textt.pack(pady=10)
        self.textt2 = tk.Label(self.root, text="Use 'threshold: 0.6 for small windows \n and 0.65 for normal windows \n windows must all have different names \n Instances must have 'MuMu Player' in the name")
        self.textt2.pack(pady=5)
        self.start_button = tk.Button(self.root, text="Start", command=self.start_script, bg="green", fg="white")
        self.start_button.pack(pady=5)
        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop_script, bg="red", fg="white",
                                     state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        self.label = tk.Label(self.root, text=f"Current threshold = {self.match_threshold}")
        self.label.pack(pady=5)

        # Threshold input
        self.entry = tk.Entry(self.root)
        self.entry.pack(pady=5)
        self.submit_button = tk.Button(self.root, text="Submit", command=self.on_submit)
        self.submit_button.pack(pady=10)

        self.label3 = tk.Label(self.root, text=f"Min matches in pack = {self.min_in_pack}")
        self.label3.pack(pady=5)

        # Matches input
        self.entry2 = tk.Entry(self.root)
        self.entry2.pack(pady=5)
        self.submit_button2 = tk.Button(self.root, text="Submit", command=self.on_submitmatches)
        self.submit_button2.pack(pady=10)

        # Status text
        self.textt3 = tk.Label(self.root, text="The finder only looks for 2* and 1* cards \n print details is slow, dont leave it on")
        self.textt3.pack(pady=10)


        self.print_details = tk.BooleanVar()
        self.checkbox = tk.Checkbutton(self.root, text="Print details", variable=self.print_details)
        self.checkbox.pack(pady=10)
        self.textt5 = tk.Label(self.root, text="")
        self.textt5.pack(pady=10)

        #self.optimize_button = tk.Button(self.root, text="Optimize Regions", command=self.optimize_regions, bg="blue", fg="white")
        #self.optimize_button.pack(pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Load MobileNetV2
        base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        self.model = Model(inputs=base_model.input, outputs=base_model.output)

    def on_close(self):
        self.stop_script()
        print("Closing application.")
        self.root.destroy()
        sys.exit(0)

    def on_submit(self):
        try:
            value = float(self.entry.get())
            if 0 <= value <= 1:
                self.label["text"] = f"Current threshold = {value}"
                self.match_threshold = value
            else:
                raise ValueError("Out of range")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a decimal number between 0 and 1.")

    def on_submitmatches(self):
        try:
            value = int(self.entry2.get())
            if 0 < value <= 5:
                self.label3["text"] = f"Min matches in pack = {value}"
                self.min_in_pack = value
            else:
                raise ValueError("Out of range")
        except ValueError:
            messagebox.showerror("Invalid Input", f"Please enter a whole number between 1 and 5.")

    def extract_features(self, image):
        """Extract features using MobileNetV2."""
        img_resized = cv2.resize(image, (112, 112))
        img_array = np.expand_dims(img_resized, axis=0)
        img_preprocessed = preprocess_input(img_array)
        features = self.model.predict(img_preprocessed)
        clear_session()
        return features.reshape(1, -1)  # Ensure features is 2D

    def load_images(self, folder_path):
        """Load images and extract features."""
        feature_templates = []
        for image_path in folder_path.glob("*.*"):
            img = cv2.imread(str(image_path))
            if img is not None:
                features = self.extract_features(img)
                feature_templates.append((image_path.name, features))
        return feature_templates

    def run(self):
        self.templates = self.load_images(IMAGES_FOLDER)
        if not self.templates:
            print(f"No images found in {IMAGES_FOLDER}.")
            sys.exit(1)
        self.root.mainloop()



    def capture_screen_in_region(self, rect, windowTitle):
        """Capture and preprocess a screen region."""
        windows = gw.getWindowsWithTitle(windowTitle)

        if not windows:
            print(f"No windows found with the title '{windowTitle}'.")
            return None

        win = windows[0]

        if win:
            left, top, width, height = win.left, win.top, win.width, win.height
            topp = top + 26.5  # Adjust for window header height if needed

            x1 = int(left + (rect[0] * width))
            y1 = int(topp + (rect[1] * height))
            x2 = int(left + (rect[2] * width))
            y2 = int(topp + (rect[3] * height))

            region_width = x2 - x1
            region_height = y2 - y1

            # Ensure the dimensions are positive and valid
            if region_width < 0:
                region_width = -region_width
            if region_height < 0:
                region_height = -region_height

            with mss() as sct:
                screenshot = sct.grab({"top": y1, "left": x1, "width": region_width, "height": region_height})
                screen_np = np.array(screenshot, dtype=np.uint8)

                # Check if the screen_np image is empty
                if screen_np.size == 0:
                    print("Captured image is empty.")
                    return None

                if screen_np.shape[2] == 4:
                    screen_np = cv2.cvtColor(screen_np, cv2.COLOR_BGRA2BGR)
                del screenshot

                # Close the mss instance and force garbage collection
                sct.close()
                gc.collect()
                return screen_np
        else:
            print("Window not found.")
            return None



    def find_match(self, screen_features):
        """Find match using cosine similarity."""

        for name, template_features in self.templates:
            sim = cosine_similarity(screen_features, template_features)
            similarity = sim[0][0]
            if name == "Wigglytuff2.png": similarity -= 0.03
            elif name == "Charmander.png": similarity -= 0.02
            elif name == "Cubone.png": similarity -= 0.03
            elif name == "Slowpoke.png": similarity += 0.05
            elif name == "Nidoqueen.png": similarity += 0.04
            elif name == "Nidoking.png": similarity += 0.04
            elif name == "Pidgeot.png": similarity += 0.04
            elif name == "Zapdos2.png": similarity += 0.03
            elif name == "Porygon.png": similarity += 0.04
            elif name == "Snorlax.png": similarity += 0.04
            elif name == "Meowth.png": similarity += 0.04
            elif name == "Pinsir.png": similarity += 0.04
            elif name == "Golbat.png": similarity += 0.05
            elif name == "Marowak.png": similarity += 0.06
            elif name == "Mewtwo.png": similarity += 0.06
            elif name == "Eevee.png": similarity += 0.06
            if similarity >= self.match_threshold:
                print(f"Match found: {name} similarity: {similarity:.3f}")
                return True
        return False


    def find_match_printed(self, screen_features, rectangle_num, window_name):
        """Find match using cosine similarity."""
        highest_similarity = 0
        highest_sim_name = ""
        rect_pos = rect_strings[rectangle_num]

        for name, template_features in self.templates:
            sim = cosine_similarity(screen_features, template_features)
            similarity = sim[0][0]
            if name == "Wigglytuff2.png": similarity -= 0.03
            elif name == "Charmander.png": similarity -= 0.02
            elif name == "Cubone.png": similarity -= 0.03
            elif name == "Slowpoke.png": similarity += 0.05
            elif name == "Nidoqueen.png": similarity += 0.04
            elif name == "Nidoking.png": similarity += 0.04
            elif name == "Pidgeot.png": similarity += 0.04
            elif name == "Zapdos2.png": similarity += 0.03
            elif name == "Porygon.png": similarity += 0.04
            elif name == "Snorlax.png": similarity += 0.04
            elif name == "Meowth.png": similarity += 0.04
            elif name == "Pinsir.png": similarity += 0.04
            elif name == "Golbat.png": similarity += 0.05
            elif name == "Marowak.png": similarity += 0.06
            elif name == "Mewtwo.png": similarity += 0.06
            elif name == "Eevee.png": similarity += 0.06
            if similarity > highest_similarity:
                #highest_similarity2 = highest_similarity
                #highest_sim_name2 = highest_sim_name
                highest_similarity = similarity
                highest_sim_name = name
            #print(f"Checking template: {name}, Similarity: {max_similarity}")
            if similarity >= self.match_threshold:
                print("/////////////////////////////////////////")
                print(f"{rect_pos} {name} with {similarity:.3f} in {window_name}")
                print(f"/////////// MATCH FOUND!!! /////////////")
               ## print(f"{name}'S VALUE {similarity:.3f} IS HIGHER THAN THE THRESHOLD")
                return True
        print(f"{rect_pos} {highest_sim_name} with {highest_similarity:.3f} in {window_name}")
        return False

    def find_closest_match(self, screen_features):
        """Find match using cosine similarity."""
        highest_similarity = 0
        highest_sim_name = ""
        for name, template_features in self.templates:
            sim = cosine_similarity(screen_features, template_features)
            similarity = sim[0][0]
            if similarity > highest_similarity:
                highest_similarity = similarity
                highest_sim_name = name
        print(f"highest similarity:{highest_sim_name} - {highest_similarity})")
        return highest_similarity

    def optimize_regions(self):
        print("Starting optimization for regions...")
        step = 0.0005  # Adjust step size for finer control
        max_adjustment = 0.005  # Allow up to ±10% adjustments in either direction
        updated_regions = []

        for idx, rect in enumerate(self.RECT_REGIONS):
            original_rect = rect[:]
            best_similarity = 0
            best_adjustment = rect[:]
            print(f"Optimizing region {idx + 1}...")

            for i in range(4):  # Iterate over each coordinate (x1, y1, x2, y2)
                direction = 1  # Start by increasing the value
                no_improvement_count = 0
                current_position = rect[i]

                while abs(direction * step) <= max_adjustment:
                    adjusted_rect = best_adjustment[:]
                    adjusted_rect[i] = max(0,
                                           min(1, current_position + direction * step))  # Clamp value between 0 and 1

                    screen_features = self.capture_screen_in_region(adjusted_rect, "MuMu Player")
                    if screen_features is None:
                        break

                    screen_features = self.extract_features(screen_features)
                    similarity = self.find_closest_match(screen_features)

                    if similarity > best_similarity:  # If improvement, update best values
                        best_similarity = similarity
                        best_adjustment = adjusted_rect[:]
                        current_position = adjusted_rect[i]
                        print(f"Improved similarity: {similarity}, Adjusted region: {adjusted_rect}")
                        no_improvement_count = 0  # Reset no improvement count
                    else:  # If no improvement, reverse direction
                        direction *= -1
                        no_improvement_count += 1
                        print(f"Similarity decreased to {similarity}. Reversing direction.")

                    # Stop adjusting this coordinate if no improvement in both directions
                    if no_improvement_count >= 2:
                        print(f"No further improvement for coordinate {i + 1} of region {idx + 1}.")
                        break

            # Validate the final improvement for the region
            if best_similarity < 0.5:  # Threshold for accepting changes
                print(f"Reverting region {idx + 1} to original due to low similarity ({best_similarity}).")
                best_adjustment = original_rect
            else:
                print(
                    f"Region {idx + 1} optimized. Best similarity: {best_similarity}, Adjusted region: {best_adjustment}")

            updated_regions.append(best_adjustment)

        # Update RECT_REGIONS with validated values
        self.RECT_REGIONS = updated_regions
        print("Optimization complete. Updated regions:", self.RECT_REGIONS)



    def main_checker(self, matching_windows):

        #tracemalloc.start()

        try:
            while self.running:
                gc.collect()
                for windowTitle in matching_windows:
                    if not self.running:
                        return
                    match_count = 0
                    miss_count = 0
                    rect_id = -1
                    for rect in self.RECT_REGIONS:
                        if not self.running:
                            return
                        rect_id += 1
                        screen_features = self.capture_screen_in_region(rect, windowTitle)
                        if screen_features is not None:
                            screen_features = self.extract_features(screen_features)
                            if self.print_details.get():
                                if self.find_match_printed(screen_features, rect_id, windowTitle):
                                    match_count += 1
                                else:
                                    miss_count += 1
                            else:
                                if self.find_match(screen_features):
                                    match_count += 1
                                else:
                                    miss_count += 1

                        if (miss_count > (5 - self.min_in_pack)) and not self.print_details.get():
                            break
                        # Check if enough matches found
                        if match_count >= self.min_in_pack:
                            self.textt5["text"] = f"GP found! Closing window: {windowTitle}"
                            try:
                                window = gw.getWindowsWithTitle(windowTitle)[0]
                                window.close()
                                time.sleep(0.5)
                            except Exception as e:
                                print(f"Error closing window '{windowTitle}': {e}")
                            #self.stop_script()
                            #time.sleep(0.5)
                            #self.start_script()
                            break
        finally:
            #cv2.destroyAllWindows()
            gc.collect()

    def start_script(self):
        if self.running:
            return
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        keyword = "MuMu Player".strip()
        matching_windows = [win for win in gw.getAllTitles() if keyword.lower() in win.lower()]

        if not matching_windows:
            print(f"No windows found with the keyword '{keyword}'.")
            self.stop_script()
            return

        threading.Thread(target=self.main_checker, args=(matching_windows,), daemon=True).start()

    def stop_script(self):
        if not self.running:
            return
        print("Stopping script...")
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        cv2.destroyAllWindows()
        gc.collect()


if __name__ == "__main__":
    app = WindowFinderApp()
    app.run()