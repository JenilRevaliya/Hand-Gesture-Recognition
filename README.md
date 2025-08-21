Hand Gesture Recognition and UI Display 🤗🎯

• This is a real-time hand gesture recognition application built with Python, using MediaPipe for hand tracking and PyQt5 for a responsive graphical user interface. The program detects various hand gestures, displays real-time information, and provides audio feedback for recognised gestures.
• The project is designed to be modular and easy to expand, making it an excellent starting point for computer vision and human-computer interaction projects.


🚀 Features

	•	Real-time Hand Tracking: Uses your webcam to detect up to two hands with high accuracy. 🎥
	•	Gesture Recognition: Recognises several predefined hand gestures, including:
	    ◦	Fist ✊
	    ◦	Open Hand 👋
	    ◦	Peace ✌️
	    ◦	Thumbs Up 👍
	    ◦	Pointing 👆
	    ◦	Pinky 🤙
	    ◦	Party 🤟
	•	Directional Detection: Identifies if the hand is pointing Left 👈 or Right 👉.
	•	Audio Feedback: A voice announces the name of a gesture when it is recognized. 🗣️
	•	Responsive GUI: Built with PyQt5, the application runs smoothly without freezing, thanks to multi-threading. 💻
	•	Clean UI Overlays: Information like FPS, gesture name, and finger count is displayed directly on the video feed. 📊


🛠️ Installation

This project requires a few key libraries. It's highly recommended to use a virtual environment to avoid conflicts with your system's Python installation. 📦
1.	Clone the Repository (or save the code): Save the provided Python script as app.py on your computer.


2.	Open Terminal and Navigate to the Project Directory:
   
    cd /path/to/your/project 


3.	Create and Activate a Virtual Environment: This step ensures all packages are installed locally for this project.

    python3.11 -m venv venv
    source venv/bin/activate

  
4.	Install the Required Libraries: The following command installs all the necessary packages, including PyQt5 for the GUI and audio libraries for voice             feedback.

   pip install mediapipe==0.10.21 opencv-python numpy PyQt5 gtts pydub simpleaudio



🏃 How to Run

After installation, simply run the Python script from your terminal with the virtual environment activated.

   python app.py


The application window will open, and your webcam feed will be displayed.
	•	Show your palm to the camera. ✋
	•	Try out different gestures like an Open Hand, Fist, or Peace sign. 🤘
	•	The application will display the detected gesture and provide a voice announcement. 🎉
	•	To exit, click the EXIT button or press q on your keyboard. 🚪


📈 Future Aspects and Expansion

This program is specifically designed to be easily expanded. Here's how you can take it further:

	•	Advanced Gesture Library: You can add more complex gestures and associate them with specific actions. The GestureRecognizer class is built for this purpose; you only need to define the new finger patterns. ✍️
 
	•	Machine Learning Integration: Instead of relying on hardcoded rules for finger patterns, you could train a custom machine learning model to recognise gestures. This would make the system more robust and adaptable to variations in hand shape and lighting. 🤖
	•	User Profiles: Implement a feature to save custom gestures for different users. This would allow a user to train the system to recognise their unique hand movements. 👤
	•	Kalman Filters: For even smoother tracking and less jitter, you could replace the current deque smoothing method with a Kalman filter. This advanced algorithm predicts the future position of a landmark based on its past movement, making it ideal for real-time tracking. Kalman Filter
