import cv2
from fer import FER
import openai
import os
from dotenv import load_dotenv


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


detector = FER(mtcnn=True)
cap = cv2.VideoCapture(0)

def get_advice_from_gpt(user_input):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": "You are a friendly and supportive assistant who helps people understand and overcome their sadness."},
                {"role": "user", "content": f"I'm feeling sad because {user_input}"}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"

sad_triggered = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detector.top_emotion(frame)
    if result:
        emotion, score = result
        cv2.putText(frame, f"{emotion} ({score:.2f})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        if emotion == "sad" and score > 0.7 and not sad_triggered:
            print("\n😢 You seem sad.")
            user_input = input("🗣️ Can you tell me what’s wrong? ")
            print("🤖 Thinking...")
            advice = get_advice_from_gpt(user_input)
            print(f"🧠 AI Advice: {advice}\n")
            sad_triggered = True  
        elif emotion != "sad":
            sad_triggered = False  

    cv2.imshow("Emotion Detection App", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
