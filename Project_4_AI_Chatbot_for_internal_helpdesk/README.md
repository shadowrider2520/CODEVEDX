# Project 4 – AI Chatbot for Internal Helpdesk

Flask-based chatbot API with intent detection, FAQ matching, and admin controls.

## Features
- Question-answer chatbot (`/chat`)
- Intent-based replies (password reset, wifi access, leave request, ticket status)
- FAQ dataset trained using TF-IDF + cosine similarity
- Admin capability to add/list/delete FAQs (API key protected)

## Setup
```bash
pip install flask pandas scikit-learn
python app.py
```
Server runs at `http://127.0.0.1:5000`

## API Endpoints

### POST /chat
Request:
```json
{ "message": "I forgot my password" }
```
Response:
```json
{ "intent": "password_reset", "answer": "Go to the IT portal..." }
```

### POST /admin/faq/add
Header: `X-Admin-Key: codevedx_admin_2026`
```json
{ "question": "How do I book a meeting room?", "answer": "Use the booking tool." }
```

### GET /admin/faq/list
Header: `X-Admin-Key: codevedx_admin_2026`

### DELETE /admin/faq/delete/<index>
Header: `X-Admin-Key: codevedx_admin_2026`

## Files
- `app.py` — main Flask app
- `helpdesk_faq.csv` — FAQ dataset

## Tech Stack
Python, Flask, pandas, scikit-learn (TF-IDF, cosine similarity)
