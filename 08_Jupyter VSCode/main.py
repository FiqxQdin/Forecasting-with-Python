import flet as ft # Flet framework
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import tempfile
from screeninfo import get_monitors  # To detect screen resolution
 
 
def main(page: ft.Page):
    # Detect screen resolution and adjust the layout for mobile devices
    screen_width = 400  # Default width
    screen_height = 300  # Default height
    try:
        monitor = get_monitors()[0]  # Get the primary monitor
        screen_width = monitor.width
        screen_height = monitor.height
    except Exception as ex:
        print(f"Failed to detect screen resolution: {str(ex)}")
 
    # Resize the page to fit the screen resolution
    page.window_width = int(screen_width * 0.8)  # 80% of screen width
    page.window_height = int(screen_height * 0.8)  # 80% of screen height
    page.scroll = "auto"  # Enable scroll if content exceeds window height
 
    # Initialize variables
    data = None
    model = None
 
    def upload_csv(e):
        nonlocal data
        if file_picker.result:
            try:
                file_path = file_picker.result.files[0].path
                data = pd.read_csv(file_path)
                result_text.value = f"Successfully loaded {file_path}"
                update_dropdowns(list(data.columns))
            except Exception as ex:
                result_text.value = f"Failed to load CSV: {str(ex)}"
        else:
            result_text.value = "No file selected"
        page.update()
 
    def update_dropdowns(columns):
        feature1_dropdown.options = [ft.dropdown.Option(col) for col in columns]
        feature2_dropdown.options = [ft.dropdown.Option(col) for col in columns]
        target_dropdown.options = [ft.dropdown.Option(col) for col in columns]
        page.update()
 
    def train_model(e):
        nonlocal model
        if data is None:
            result_text.value = "Please upload or paste data first"
            page.update()
            return
 
        feature1 = feature1_dropdown.value
        feature2 = feature2_dropdown.value
        target = target_dropdown.value
 
        if not feature1 or not feature2 or not target:
            result_text.value = "Please select features and target"
            page.update()
            return
 
        if feature1 == feature2:
            result_text.value = "Feature 1 and Feature 2 must be different"
            page.update()
            return
 
        try:
            X = data[[feature1, feature2]].values
            y = data[target].values
            model = LogisticRegression()
            model.fit(X, y)
            result_text.value = "Model trained successfully!"
        except Exception as ex:
            result_text.value = f"Error while training model: {str(ex)}"
        page.update()
 
    def view_graph(e):
        if data is None:
            result_text.value = "Please upload or paste data first"
            page.update()
            return
 
        feature1 = feature1_dropdown.value
        feature2 = feature2_dropdown.value
        target = target_dropdown.value
 
        if not feature1 or not feature2 or not target:
            result_text.value = "Please select features and target"
            page.update()
            return
 
        try:
            plt.figure(figsize=(8, 6))
            plt.scatter(data[feature1], data[feature2], c=data[target], cmap='viridis', edgecolor='k')
            plt.colorbar(label=target)
            plt.xlabel(feature1)
            plt.ylabel(feature2)
            plt.title(f'{feature1} vs {feature2} (Target: {target})')
 
            # Save and display the plot
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            plt.savefig(temp_file.name)
            plt.close()
 
            image_view.src = temp_file.name
        except Exception as ex:
            result_text.value = f"Error while plotting graph: {str(ex)}"
        page.update()
 
    def predict(e):
        if model is None:
            result_text.value = "Please train the model first"
            page.update()
            return
 
        try:
            input1 = float(feature1_input.value)
            input2 = float(feature2_input.value)
            prediction = model.predict([[input1, input2]])[0]
            result_text.value = f"Prediction: {prediction}"
        except ValueError:
            result_text.value = "Invalid input. Please enter valid numbers."
        except Exception as ex:
            result_text.value = f"Error during prediction: {str(ex)}"
        page.update()
 
    # UI Components
    file_picker = ft.FilePicker(on_result=upload_csv)
    page.overlay.append(file_picker)
 
    title = ft.Text("Predict Anything App", size=20, weight="bold")
 
    upload_button = ft.ElevatedButton("Upload CSV", on_click=lambda _: file_picker.pick_files(allow_multiple=False))
 
    feature1_dropdown = ft.Dropdown(label="Feature 1", options=[])
    feature2_dropdown = ft.Dropdown(label="Feature 2", options=[])
    target_dropdown = ft.Dropdown(label="Target", options=[])
 
    train_button = ft.ElevatedButton("Train Model", on_click=train_model)
 
    feature1_input = ft.TextField(label="Enter value for Feature 1")
    feature2_input = ft.TextField(label="Enter value for Feature 2")
 
    predict_button = ft.ElevatedButton("Make Prediction", on_click=predict)
 
    view_graph_button = ft.ElevatedButton("View Graph", on_click=view_graph)
 
    image_view = ft.Image()
 
    result_text = ft.Text("")
 
    # Layout: Use a column for vertical stacking on smaller screens (smartphone view)
    page.add(
        title,
        upload_button,
        feature1_dropdown,
        feature2_dropdown,
        target_dropdown,
        train_button,
        feature1_input,
        feature2_input,
        predict_button,
        view_graph_button,
        image_view,
        result_text,
    )
 
    # Update layout based on screen size
    if screen_width < 400:  # Small screens like smartphones
        page.add(
            ft.Column(
                controls=[
                    title,
                    upload_button,
                    feature1_dropdown,
                    feature2_dropdown,
                    target_dropdown,
                    train_button,
                    feature1_input,
                    feature2_input,
                    predict_button,
                    view_graph_button,
                    image_view,
                    result_text,
                ],
                alignment="center",  # Center elements vertically
                spacing=10,  # Add space between elements
            )
        )
 
 
# Run the Flet app
ft.app(target=main)