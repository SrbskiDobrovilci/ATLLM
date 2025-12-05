// DOM Elements
const messageForm = document.getElementById('messageForm');
const messageInput = document.getElementById('messageInput');
const chatMessages = document.getElementById('chatMessages');
const conversationsList = document.getElementById('conversationsList');
const newChatBtn = document.getElementById('newChatBtn');
const editTitleBtn = document.getElementById('editTitleBtn');
const deleteChatBtn = document.getElementById('deleteChatBtn');
const conversationTitle = document.getElementById('conversationTitle');
const editTitleModal = document.getElementById('editTitleModal');
const newTitleInput = document.getElementById('newTitleInput');
const saveTitleBtn = document.getElementById('saveTitleBtn');
const cancelEditBtn = document.getElementById('cancelEditBtn');
const charCount = document.getElementById('charCount');
const chatCount = document.getElementById('chatCount');

// Global variables
let currentConversationId = null;
let conversations = [];
let isTyping = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadConversations();
    setupEventListeners();
    adjustTextareaHeight();
});

// Event Listeners
function setupEventListeners() {
    messageForm.addEventListener('submit', handleMessageSubmit);
    messageInput.addEventListener('input', handleInputChange);
    messageInput.addEventListener('keydown', handleKeyDown);
    newChatBtn.addEventListener('click', createNewConversation);
    editTitleBtn.addEventListener('click', showEditTitleModal);
    deleteChatBtn.addEventListener('click', deleteCurrentConversation);
    saveTitleBtn.addEventListener('click', saveConversationTitle);
    cancelEditBtn.addEventListener('click', hideEditTitleModal);
    
    // Close modal when clicking outside
    editTitleModal.addEventListener('click', (e) => {
        if (e.target === editTitleModal) {
            hideEditTitleModal();
        }
    });
}

// Handle message submission
async function handleMessageSubmit(e) {
    e.preventDefault();
    
    const message = messageInput.value.trim();
    if (!message || isTyping) return;
    
    // Add user message to chat
    addMessageToChat('user', message);
    messageInput.value = '';
    adjustTextareaHeight();
    charCount.textContent = '0';
    
    // Show typing indicator
    showTypingIndicator();
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                conversation_id: currentConversationId
            })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator();
        
        if (response.ok) {
            // Add assistant response to chat
            addMessageToChat('assistant', data.response);
            
            // Update conversation list if new conversation
            if (!currentConversationId) {
                currentConversationId = data.conversation_id;
                loadConversations();
            }
        } else {
            addErrorMessage(data.error || 'Failed to get response');
        }
    } catch (error) {
        console.error('Error:', error);
        removeTypingIndicator();
        addErrorMessage('Network error. Please try again.');
    }
}

// Handle input changes
function handleInputChange() {
    adjustTextareaHeight();
    charCount.textContent = messageInput.value.length;
}

// Handle Enter key (Ctrl+Enter for new line)
function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey) {
        e.preventDefault();
        if (messageInput.value.trim()) {
            messageForm.dispatchEvent(new Event('submit'));
        }
    }
}

// Adjust textarea height based on content
function adjustTextareaHeight() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 150) + 'px';
}

// Add message to chat display
function addMessageToChat(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    // Add header with role icon
    const header = document.createElement('div');
    header.className = 'message-header';
    
    const icon = document.createElement('i');
    icon.className = role === 'user' ? 'fas fa-user' : 'fas fa-robot';
    header.appendChild(icon);
    
    const roleText = document.createElement('span');
    roleText.textContent = role === 'user' ? 'You' : 'GigaChat';
    header.appendChild(roleText);
    
    messageContent.appendChild(header);
    
    // Add content
    const contentDiv = document.createElement('div');
    contentDiv.textContent = content;
    messageContent.appendChild(contentDiv);
    
    // Add timestamp
    const timestamp = document.createElement('div');
    timestamp.className = 'message-timestamp';
    timestamp.textContent = new Date().toLocaleTimeString([], { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
    messageContent.appendChild(timestamp);
    
    messageDiv.appendChild(messageContent);
    
    // If welcome message exists, remove it
    const welcomeMessage = document.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Add error message
function addErrorMessage(error) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'message assistant error';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = `
        <div class="message-header">
            <i class="fas fa-exclamation-circle"></i>
            <span>Error</span>
        </div>
        <div>${error}</div>
    `;
    
    errorDiv.appendChild(contentDiv);
    chatMessages.appendChild(errorDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Show typing indicator
function showTypingIndicator() {
    isTyping = true;
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.id = 'typingIndicator';
    
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('div');
        dot.className = 'typing-dot';
        typingDiv.appendChild(dot);
    }
    
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Remove typing indicator
function removeTypingIndicator() {
    isTyping = false;
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Load conversations
async function loadConversations() {
    try {
        const response = await fetch('/api/conversations');
        const data = await response.json();
        
        if (response.ok) {
            conversations = data;
            renderConversationsList();
            updateChatCounter();
            
            // Load first conversation if none selected
            if (!currentConversationId && data.length > 0) {
                loadConversation(data[0].id);
            }
        }
    } catch (error) {
        console.error('Error loading conversations:', error);
    }
}

// Render conversations list
function renderConversationsList() {
    conversationsList.innerHTML = '';
    
    conversations.forEach(conv => {
        const convItem = document.createElement('div');
        convItem.className = `conversation-item ${conv.id === currentConversationId ? 'active' : ''}`;
        convItem.dataset.conversationId = conv.id;
        
        const titleSpan = document.createElement('span');
        titleSpan.className = 'conversation-title';
        titleSpan.textContent = conv.title || 'New Conversation';
        
        const dateSpan = document.createElement('span');
        dateSpan.className = 'conversation-date';
        dateSpan.textContent = formatDate(conv.updated_at);
        
        convItem.appendChild(titleSpan);
        convItem.appendChild(dateSpan);
        
        convItem.addEventListener('click', () => loadConversation(conv.id));
        
        conversationsList.appendChild(convItem);
    });
}

// Load specific conversation
async function loadConversation(conversationId) {
    try {
        const response = await fetch(`/api/conversations/${conversationId}`);
        const data = await response.json();
        
        if (response.ok) {
            currentConversationId = conversationId;
            conversationTitle.textContent = data.title || 'New Conversation';
            
            // Clear chat messages
            chatMessages.innerHTML = '';
            
            // Add messages to chat
            if (data.messages && data.messages.length > 0) {
                data.messages.forEach(msg => {
                    addMessageToChat(msg.role, msg.content);
                });
            } else {
                // Show welcome message if no messages
                const welcomeMessage = document.querySelector('.welcome-message');
                if (!welcomeMessage) {
                    chatMessages.innerHTML = `
                        <div class="welcome-message">
                            <h3>Conversation loaded</h3>
                            <p>Start chatting with GigaChat!</p>
                        </div>
                    `;
                }
            }
            
            // Update active conversation in list
            renderConversationsList();
        }
    } catch (error) {
        console.error('Error loading conversation:', error);
    }
}

// Create new conversation
function createNewConversation() {
    currentConversationId = null;
    conversationTitle.textContent = 'New Conversation';
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <h3>Welcome to GigaChat!</h3>
            <p>Start a new conversation by typing a message below.</p>
            <div class="example-prompts">
                <p>Try asking:</p>
                <ul>
                    <li>"Explain quantum computing in simple terms"</li>
                    <li>"Write a Python function to sort a list"</li>
                    <li>"Help me plan a trip to Japan"</li>
                </ul>
            </div>
        </div>
    `;
    
    // Update active conversation in list
    renderConversationsList();
    messageInput.focus();
}

// Show edit title modal
function showEditTitleModal() {
    if (!currentConversationId) {
        alert('Please start a conversation first');
        return;
    }
    
    newTitleInput.value = conversationTitle.textContent;
    editTitleModal.style.display = 'flex';
    newTitleInput.focus();
}

// Hide edit title modal
function hideEditTitleModal() {
    editTitleModal.style.display = 'none';
}

// Save conversation title
async function saveConversationTitle() {
    const newTitle = newTitleInput.value.trim();
    
    if (!newTitle) {
        alert('Title cannot be empty');
        return;
    }
    
    try {
        const response = await fetch(`/api/conversations/${currentConversationId}/title`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ title: newTitle })
        });
        
        if (response.ok) {
            conversationTitle.textContent = newTitle;
            hideEditTitleModal();
            loadConversations(); // Refresh list
        } else {
            const data = await response.json();
            alert(data.error || 'Failed to update title');
        }
    } catch (error) {
        console.error('Error updating title:', error);
        alert('Network error');
    }
}

// Delete current conversation
async function deleteCurrentConversation() {
    if (!currentConversationId) {
        alert('No conversation to delete');
        return;
    }
    
    if (!confirm('Are you sure you want to delete this conversation?')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/conversations/${currentConversationId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            createNewConversation();
            loadConversations();
        } else {
            const data = await response.json();
            alert(data.error || 'Failed to delete conversation');
        }
    } catch (error) {
        console.error('Error deleting conversation:', error);
        alert('Network error');
    }
}

// Update chat counter
function updateChatCounter() {
    chatCount.textContent = conversations.length;
}

// Format date
function formatDate(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    if (diffMins < 60) {
        return `${diffMins}m ago`;
    } else if (diffHours < 24) {
        return `${diffHours}h ago`;
    } else if (diffDays < 7) {
        return `${diffDays}d ago`;
    } else {
        return date.toLocaleDateString();
    }
}

// Example prompt click handler
document.addEventListener('click', (e) => {
    if (e.target.closest('.example-prompts li')) {
        const prompt = e.target.closest('li').textContent;
        messageInput.value = prompt;
        adjustTextareaHeight();
        charCount.textContent = prompt.length;
        messageInput.focus();
    }
});