import os
import uuid
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import requests
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    conversations = db.relationship('Conversation', backref='user', lazy=True, cascade='all, delete-orphan')

class Conversation(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), default='New Conversation')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    messages = db.relationship('Message', backref='conversation', lazy=True, cascade='all, delete-orphan')

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.String(36), db.ForeignKey('conversation.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    tokens = db.Column(db.Integer, default=0)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# GigaChat API Integration
class GigaChatClient:
    def __init__(self, client_id, client_secret, scope, auth_url, api_url):
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self.auth_url = auth_url
        self.api_url = api_url
        self.access_token = None
        self.token_expires = None
    
    def get_access_token(self):
        """Get access token for GigaChat API"""
        try:
            rq_headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json',
                'RqUID': str(uuid.uuid4()),
            }
            
            rq_body = {
                'scope': self.scope
            }
            
            response = requests.post(
                self.auth_url,
                headers=rq_headers,
                data=rq_body,
                auth=(self.client_id, self.client_secret),
                verify=False  # Only for development with self-signed certs
            )
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data['access_token']
                return self.access_token
            else:
                print(f"Token request failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error getting access token: {e}")
            return None
    
    def send_message(self, messages, max_tokens=1024, temperature=0.7):
        """Send messages to GigaChat and get response"""
        if not self.access_token:
            if not self.get_access_token():
                return None
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': 'GigaChat',
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=payload,
                verify=False
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:  # Token expired
                self.get_access_token()
                headers['Authorization'] = f'Bearer {self.access_token}'
                response = requests.post(
                    f"{self.api_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    verify=False
                )
                if response.status_code == 200:
                    return response.json()
            
            print(f"API request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
        except Exception as e:
            print(f"Error sending message: {e}")
            return None

# Initialize GigaChat client
giga_client = GigaChatClient(
    client_id=app.config['GIGACHAT_CLIENT_ID'],
    client_secret=app.config['GIGACHAT_CLIENT_SECRET'],
    scope=app.config['GIGACHAT_SCOPE'],
    auth_url=app.config['GIGACHAT_AUTH_URL'],
    api_url=app.config['GIGACHAT_API_URL']
)

# Routes
@app.route('/')
@login_required
def index():
    conversations = Conversation.query.filter_by(user_id=current_user.id)\
        .order_by(Conversation.updated_at.desc())\
        .limit(app.config['MAX_CONTEXT_CHATS'])\
        .all()
    
    # Get recent messages for the latest conversation
    recent_messages = []
    if conversations:
        latest_conversation = conversations[0]
        recent_messages = Message.query.filter_by(conversation_id=latest_conversation.id)\
            .order_by(Message.timestamp.asc())\
            .limit(20)\
            .all()
    
    return render_template('index.html', 
                         conversations=conversations, 
                         messages=recent_messages,
                         max_chats=app.config['MAX_CONTEXT_CHATS'])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('index'))
        
        return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            return render_template('register.html', error='Username already exists')
        
        user = User(
            username=username,
            password_hash=generate_password_hash(password)
        )
        
        db.session.add(user)
        db.session.commit()
        
        login_user(user)
        return redirect(url_for('index'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    data = request.json
    message = data.get('message')
    conversation_id = data.get('conversation_id')
    
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    
    # Get or create conversation
    if conversation_id:
        conversation = Conversation.query.filter_by(
            id=conversation_id, 
            user_id=current_user.id
        ).first()
        if not conversation:
            return jsonify({'error': 'Conversation not found'}), 404
    else:
        # Create new conversation
        conversation = Conversation(
            user_id=current_user.id,
            title=message[:100] + ('...' if len(message) > 100 else '')
        )
        db.session.add(conversation)
        db.session.commit()
    
    # Get recent messages for context (last 10 messages)
    previous_messages = Message.query.filter_by(conversation_id=conversation.id)\
        .order_by(Message.timestamp.desc())\
        .limit(10)\
        .all()
    
    previous_messages.reverse()  # Oldest first
    
    # Format messages for GigaChat API
    messages_for_api = []
    for msg in previous_messages:
        messages_for_api.append({
            'role': msg.role,
            'content': msg.content
        })
    
    # Add current user message
    messages_for_api.append({
        'role': 'user',
        'content': message
    })
    
    # Save user message
    user_message = Message(
        conversation_id=conversation.id,
        role='user',
        content=message
    )
    db.session.add(user_message)
    
    # Get response from GigaChat
    response_data = giga_client.send_message(messages_for_api)
    
    if response_data and 'choices' in response_data and len(response_data['choices']) > 0:
        assistant_message = response_data['choices'][0]['message']['content']
        
        # Save assistant response
        assistant_msg = Message(
            conversation_id=conversation.id,
            role='assistant',
            content=assistant_message,
            tokens=response_data.get('usage', {}).get('total_tokens', 0)
        )
        db.session.add(assistant_msg)
        
        # Update conversation timestamp
        conversation.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'response': assistant_message,
            'conversation_id': conversation.id,
            'message_id': assistant_msg.id
        })
    else:
        return jsonify({'error': 'Failed to get response from GigaChat'}), 500

@app.route('/api/conversations', methods=['GET'])
@login_required
def get_conversations():
    conversations = Conversation.query.filter_by(user_id=current_user.id)\
        .order_by(Conversation.updated_at.desc())\
        .limit(app.config['MAX_CONTEXT_CHATS'])\
        .all()
    
    conversations_list = []
    for conv in conversations:
        conversations_list.append({
            'id': conv.id,
            'title': conv.title,
            'created_at': conv.created_at.isoformat(),
            'updated_at': conv.updated_at.isoformat(),
            'message_count': len(conv.messages)
        })
    
    return jsonify(conversations_list)

@app.route('/api/conversations/<conversation_id>', methods=['GET'])
@login_required
def get_conversation(conversation_id):
    conversation = Conversation.query.filter_by(
        id=conversation_id, 
        user_id=current_user.id
    ).first()
    
    if not conversation:
        return jsonify({'error': 'Conversation not found'}), 404
    
    messages = Message.query.filter_by(conversation_id=conversation.id)\
        .order_by(Message.timestamp.asc())\
        .all()
    
    messages_list = []
    for msg in messages:
        messages_list.append({
            'id': msg.id,
            'role': msg.role,
            'content': msg.content,
            'timestamp': msg.timestamp.isoformat()
        })
    
    return jsonify({
        'id': conversation.id,
        'title': conversation.title,
        'messages': messages_list
    })

@app.route('/api/conversations/<conversation_id>', methods=['DELETE'])
@login_required
def delete_conversation(conversation_id):
    conversation = Conversation.query.filter_by(
        id=conversation_id, 
        user_id=current_user.id
    ).first()
    
    if not conversation:
        return jsonify({'error': 'Conversation not found'}), 404
    
    db.session.delete(conversation)
    db.session.commit()
    
    return jsonify({'success': True})

@app.route('/api/conversations/<conversation_id>/title', methods=['PUT'])
@login_required
def update_conversation_title(conversation_id):
    data = request.json
    new_title = data.get('title', '').strip()
    
    if not new_title:
        return jsonify({'error': 'Title is required'}), 400
    
    conversation = Conversation.query.filter_by(
        id=conversation_id, 
        user_id=current_user.id
    ).first()
    
    if not conversation:
        return jsonify({'error': 'Conversation not found'}), 404
    
    conversation.title = new_title[:200]
    db.session.commit()
    
    return jsonify({'success': True, 'title': conversation.title})

# Initialize database
with app.app_context():
    db.create_all()
    
    # Create admin user if not exists (for testing)
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            password_hash=generate_password_hash('admin123')
        )
        db.session.add(admin)
        db.session.commit()

if __name__ == '__main__':
    app.run(debug=True, port=5000)