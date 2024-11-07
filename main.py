from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import numpy as np
import random

app = Flask(__name__)

# MediaPipe Hand Detection Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Game settings
WIDTH, HEIGHT = 640, 480
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 100
BALL_SIZE = 20
BALL_SPEED_X, BALL_SPEED_Y = 30, 30
WINNING_SCORE = 5

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

# Initialize paddles and ball
left_paddle_y = HEIGHT // 2 - PADDLE_HEIGHT // 2
right_paddle_y = HEIGHT // 2 - PADDLE_HEIGHT // 2
ball_x, ball_y = WIDTH // 2, HEIGHT // 2

# Initialize scores
left_score = 0
right_score = 0
game_over = False
winner = ""
game_started = False  # Game start flag


# Video streaming function
def generate_frames():
    global ball_x, ball_y, BALL_SPEED_X, BALL_SPEED_Y, left_paddle_y, right_paddle_y, left_score, right_score, game_over, winner, game_started

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if game_started and not game_over:
            # Hand tracking
            results = hands.process(frame_rgb)

            # Initialize paddle movements
            left_paddle_y_new = None
            right_paddle_y_new = None

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = handedness.classification[0].label.lower()  # 'left' or 'right'
                    if hand_label == 'left':
                        left_paddle_y_new = update_paddle_y(hand_landmarks, 'left', frame)
                    elif hand_label == 'right':
                        right_paddle_y_new = update_paddle_y(hand_landmarks, 'right', frame)

                    # Draw hand landmarks for visualization
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Update paddle positions if detected
            if left_paddle_y_new is not None:
                left_paddle_y = np.clip(left_paddle_y_new, 0, HEIGHT - PADDLE_HEIGHT)
            if right_paddle_y_new is not None:
                right_paddle_y = np.clip(right_paddle_y_new, 0, HEIGHT - PADDLE_HEIGHT)

            # Move ball
            ball_movement()

            # Check for paddle collisions
            paddle_ball_collision(left_paddle_y, right_paddle_y)

            # Check for winning condition
            if left_score >= WINNING_SCORE:
                game_over = True
                winner = "Left Player"
            elif right_score >= WINNING_SCORE:
                game_over = True
                winner = "Right Player"

            # Draw paddles, ball, and scores
            draw_paddles_and_ball(frame, left_paddle_y, right_paddle_y, ball_x, ball_y)
            draw_scores(frame, left_score, right_score)
        elif game_started and game_over:
            # Display game over and winner
            draw_game_over(frame, winner)
        frame = cv2.resize(frame, (0, 0), fx=1.85, fy=1.3)
        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_game', methods=['POST'])
def start_game():
    global game_started, game_over, left_score, right_score
    game_started = True
    game_over = False
    left_score = 0
    right_score = 0
    return '', 204


def draw_paddles_and_ball(frame, left_paddle_y, right_paddle_y, ball_x, ball_y):
    cv2.rectangle(frame, (10, left_paddle_y), (10 + PADDLE_WIDTH, left_paddle_y + PADDLE_HEIGHT), WHITE, -1)
    cv2.rectangle(frame, (WIDTH - 20, right_paddle_y), (WIDTH - 20 + PADDLE_WIDTH, right_paddle_y + PADDLE_HEIGHT),
                  WHITE, -1)
    cv2.circle(frame, (ball_x, ball_y), BALL_SIZE // 2, GREEN, -1)


def draw_scores(frame, left_score, right_score):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f'Left: {left_score}', (50, 50), font, 1, WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, f'Right: {right_score}', (WIDTH - 200, 50), font, 1, WHITE, 2, cv2.LINE_AA)


def draw_game_over(frame, winner):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Game Over", (WIDTH // 4, HEIGHT // 2 - 40), font, 2, RED, 3, cv2.LINE_AA)
    cv2.putText(frame, f"{winner} wins!", (WIDTH // 4, HEIGHT // 2 + 20), font, 1, RED, 2, cv2.LINE_AA)


def ball_movement():
    global ball_x, ball_y, BALL_SPEED_X, BALL_SPEED_Y, left_score, right_score
    ball_x += BALL_SPEED_X
    ball_y += BALL_SPEED_Y

    if ball_y - BALL_SIZE // 2 <= 0 or ball_y + BALL_SIZE // 2 >= HEIGHT:
        BALL_SPEED_Y = -BALL_SPEED_Y

    if ball_x - BALL_SIZE // 2 <= 0:
        right_score += 1
        reset_ball()
    elif ball_x + BALL_SIZE // 2 >= WIDTH:
        left_score += 1
        reset_ball()


def reset_ball():
    global ball_x, ball_y, BALL_SPEED_X
    ball_x, ball_y = WIDTH // 2, HEIGHT // 2
    BALL_SPEED_X = random.choice([-10, 10])


def paddle_ball_collision(left_paddle_y, right_paddle_y):
    global BALL_SPEED_X
    if ball_x - BALL_SIZE // 2 <= 20 and left_paddle_y <= ball_y <= left_paddle_y + PADDLE_HEIGHT:
        BALL_SPEED_X = -BALL_SPEED_X
    if ball_x + BALL_SIZE // 2 >= WIDTH - 20 and right_paddle_y <= ball_y <= right_paddle_y + PADDLE_HEIGHT:
        BALL_SPEED_X = -BALL_SPEED_X


def update_paddle_y(hand_landmarks, hand_side, frame):
    if hand_landmarks:
        for id, lm in enumerate(hand_landmarks.landmark):
            h, w, _ = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            if id == 8:  # Index finger tip
                return cy - PADDLE_HEIGHT // 2
    return None


if __name__ == "__main__":
    app.run(port=80,debug=True)



