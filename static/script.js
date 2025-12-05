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

// Add these functions to the existing script.js

// RAG-related functions
let currentRagEnabled = true;
let currentContextSources = [];

// Initialize RAG
function initializeRAG() {
    // Load RAG status
    fetch('/api/rag/info')
        .then(response => response.json())
        .then(data => {
            if (data.enabled) {
                updateRAGStatusUI(data);
            } else {
                document.getElementById('ragStatusBadge').style.display = 'none';
                document.getElementById('ragToggleContainer').style.display = 'none';
            }
        })
        .catch(error => {
            console.error('Error loading RAG info:', error);
        });
    
    // Setup RAG toggle
    const ragToggle = document.getElementById('ragToggle');
    if (ragToggle) {
        ragToggle.addEventListener('change', toggleRAG);
    }
    
    // Setup search button
    const searchDocsBtn = document.getElementById('searchDocsBtn');
    if (searchDocsBtn) {
        searchDocsBtn.addEventListener('click', showSearchModal);
    }
    
    // Setup context sources button
    const showSourcesBtn = document.getElementById('showSourcesBtn');
    if (showSourcesBtn) {
        showSourcesBtn.addEventListener('click', toggleContextSources);
    }
    
    // Setup clear context button
    const clearContextBtn = document.getElementById('clearContextBtn');
    if (clearContextBtn) {
        clearContextBtn.addEventListener('click', clearContext);
    }
}

// Update RAG status UI
function updateRAGStatusUI(ragInfo) {
    const ragBadge = document.getElementById('ragStatusBadge');
    const ragToggleContainer = document.getElementById('ragToggleContainer');
    
    if (ragBadge && ragToggleContainer) {
        ragToggleContainer.style.display = 'flex';
        
        if (ragInfo.total_chunks > 0) {
            ragBadge.innerHTML = `<i class="fas fa-database"></i> ${ragInfo.total_chunks} документов`;
        }
    }
}

// Toggle RAG for current conversation
function toggleRAG() {
    const ragToggle = document.getElementById('ragToggle');
    currentRagEnabled = ragToggle.checked;
    
    if (currentConversationId) {
        fetch(`/api/conversations/${currentConversationId}/toggle_rag`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showNotification(
                    currentRagEnabled ? 
                    'Legal context enabled' : 
                    'Legal context disabled',
                    currentRagEnabled ? 'success' : 'info'
                );
            }
        })
        .catch(error => {
            console.error('Error toggling RAG:', error);
        });
    }
}

// Show search modal
function showSearchModal() {
    const searchModal = document.getElementById('searchModal');
    searchModal.style.display = 'flex';
    
    const searchInput = document.getElementById('docSearchInput');
    searchInput.focus();
    
    // Setup search button
    const performSearchBtn = document.getElementById('performSearchBtn');
    const closeSearchBtn = document.getElementById('closeSearchBtn');
    
    performSearchBtn.onclick = performDocumentSearch;
    closeSearchBtn.onclick = () => {
        searchModal.style.display = 'none';
    };
    
    // Close modal when clicking outside
    searchModal.onclick = (e) => {
        if (e.target === searchModal) {
            searchModal.style.display = 'none';
        }
    };
    
    // Enter key for search
    searchInput.onkeydown = (e) => {
        if (e.key === 'Enter') {
            performDocumentSearch();
        }
    };
}

// Perform document search
function performDocumentSearch() {
    const searchInput = document.getElementById('docSearchInput');
    const query = searchInput.value.trim();
    
    if (!query) return;
    
    fetch('/api/rag/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query: query,
            k: 10
        })
    })
    .then(response => response.json())
    .then(data => {
        displaySearchResults(data.results);
    })
    .catch(error => {
        console.error('Search error:', error);
        document.getElementById('searchResults').innerHTML = `
            <div class="error-message">
                Error performing search. Please try again.
            </div>
        `;
    });
}

// Display search results
function displaySearchResults(results) {
    const searchResults = document.getElementById('searchResults');
    
    if (!results || results.length === 0) {
        searchResults.innerHTML = `
            <div class="no-results">
                <i class="fas fa-search"></i>
                <p>No matching documents found.</p>
            </div>
        `;
        return;
    }
    
    let html = '';
    
    results.forEach((result, index) => {
        const chunk = result.chunk;
        const metadata = result.metadata;
        const similarity = (result.similarity_score * 100).toFixed(1);
        
        html += `
            <div class="search-result-item">
                <div class="search-result-header">
                    <div class="search-result-title">
                        ${metadata.source || 'Unknown Document'}
                        <span class="doc-type-badge">${metadata.document_type || 'legal'}</span>
                    </div>
                    <div class="search-result-score">
                        ${similarity}% match
                    </div>
                </div>
                <div class="search-result-content">
                    ${truncateText(chunk, 300)}
                </div>
                <div class="search-result-meta">
                    <span>${metadata.text_length || chunk.length} chars</span>
                    <span>Use in chat: 
                        <button class="use-in-chat-btn" data-chunk="${encodeURIComponent(chunk)}">
                            <i class="fas fa-comment-alt"></i>
                        </button>
                    </span>
                </div>
            </div>
        `;
    });
    
    searchResults.innerHTML = html;
    
    // Add event listeners for use-in-chat buttons
    document.querySelectorAll('.use-in-chat-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const chunk = decodeURIComponent(this.dataset.chunk);
            useChunkInChat(chunk);
        });
    });
}

// Use search result in chat
function useChunkInChat(chunk) {
    const messageInput = document.getElementById('messageInput');
    messageInput.value = `Based on this legal text:\n\n"${truncateText(chunk, 200)}"\n\n`;
    adjustTextareaHeight();
    charCount.textContent = messageInput.value.length;
    
    // Close search modal
    document.getElementById('searchModal').style.display = 'none';
    messageInput.focus();
}

// Toggle context sources panel
function toggleContextSources() {
    const panel = document.getElementById('contextSourcesPanel');
    const isVisible = panel.style.display === 'block';
    
    if (isVisible) {
        panel.style.display = 'none';
    } else {
        panel.style.display = 'block';
        updateContextSourcesDisplay();
    }
}

// Update context sources display
function updateContextSourcesDisplay() {
    const sourcesList = document.getElementById('sourcesList');
    
    if (!currentContextSources || currentContextSources.length === 0) {
        sourcesList.innerHTML = `
            <div class="no-sources">
                <i class="fas fa-info-circle"></i>
                <p>No legal references used in this response.</p>
            </div>
        `;
        return;
    }
    
    let html = '';
    
    currentContextSources.forEach((source, index) => {
        html += `
            <div class="source-item">
                <div class="source-header">
                    <div class="source-title">
                        ${source.document_type || 'Legal Document'}
                    </div>
                    <div class="source-confidence">
                        ${(source.similarity * 100).toFixed(0)}% relevant
                    </div>
                </div>
                <div class="source-meta">
                    Source: ${source.source || 'Unknown'}
                </div>
            </div>
        `;
    });
    
    sourcesList.innerHTML = html;
}

// Clear context for current conversation
function clearContext() {
    if (currentConversationId) {
        // Implementation would depend on backend API
        showNotification('Context cleared for this conversation', 'info');
    } else {
        showNotification('Start a conversation first', 'warning');
    }
}

// Update handleMessageSubmit to include RAG
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
                conversation_id: currentConversationId,
                use_rag: currentRagEnabled
            })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator();
        
        if (response.ok) {
            // Add assistant response to chat with context indicator
            addMessageToChat('assistant', data.response, data.context_used, data.context_sources);
            
            // Store context sources for display
            currentContextSources = data.context_sources || [];
            
            // Update conversation list if new conversation
            if (!currentConversationId) {
                currentConversationId = data.conversation_id;
                loadConversations();
                
                // Show RAG toggle for new conversation
                const ragToggleContainer = document.getElementById('ragToggleContainer');
                if (ragToggleContainer) {
                    ragToggleContainer.style.display = 'flex';
                }
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

// Update addMessageToChat to handle context
function addMessageToChat(role, content, contextUsed = false, contextSources = []) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    if (contextUsed) {
        messageDiv.classList.add('context-used');
    }
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    // Add header with role icon
    const header = document.createElement('div');
    header.className = 'message-header';
    
    const icon = document.createElement('i');
    icon.className = role === 'user' ? 'fas fa-user' : 'fas fa-robot';
    header.appendChild(icon);
    
    const roleText = document.createElement('span');
    roleText.textContent = role === 'user' ? 'You' : 'Legal Assistant';
    header.appendChild(roleText);
    
    // Add context indicator for assistant messages with context
    if (role === 'assistant' && contextUsed) {
        const contextIndicator = document.createElement('span');
        contextIndicator.className = 'context-indicator';
        contextIndicator.innerHTML = `
            <i class="fas fa-file-contract"></i>
            Based on legal documents
            ${contextSources.length > 0 ? 
                `<button class="context-sources-btn" data-sources='${JSON.stringify(contextSources)}'>
                    (${contextSources.length} sources)
                </button>` : 
                ''}
        `;
        header.appendChild(contextIndicator);
    }
    
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
    
    // Add click handler for context sources button
    if (contextSources.length > 0) {
        const sourcesBtn = messageDiv.querySelector('.context-sources-btn');
        if (sourcesBtn) {
            sourcesBtn.addEventListener('click', function() {
                const sources = JSON.parse(this.dataset.sources);
                showContextSourcesTooltip(sources, this);
            });
        }
    }
}

// Show context sources tooltip
function showContextSourcesTooltip(sources, element) {
    const tooltip = document.getElementById('contextTooltip');
    
    let html = '<div class="tooltip-content">';
    html += '<strong>Legal References:</strong><ul>';
    
    sources.forEach(source => {
        html += `<li>
            <strong>${source.document_type || 'Document'}:</strong> 
            ${source.source || 'Unknown'}
            (${(source.similarity * 100).toFixed(0)}% relevant)
        </li>`;
    });
    
    html += '</ul></div>';
    
    tooltip.innerHTML = html;
    tooltip.style.display = 'block';
    
    // Position tooltip near the element
    const rect = element.getBoundingClientRect();
    tooltip.style.left = `${rect.left}px`;
    tooltip.style.top = `${rect.bottom + 5}px`;
    
    // Hide tooltip after 5 seconds or on click
    setTimeout(() => {
        tooltip.style.display = 'none';
    }, 5000);
    
    // Also hide on click
    document.addEventListener('click', function hideTooltip() {
        tooltip.style.display = 'none';
        document.removeEventListener('click', hideTooltip);
    });
}

// Show notification
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Show with animation
    setTimeout(() => {
        notification.classList.add('show');
    }, 10);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 3000);
}

// Truncate text
function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substr(0, maxLength) + '...';
}

// Update loadConversation to handle RAG toggle
async function loadConversation(conversationId) {
    try {
        const response = await fetch(`/api/conversations/${conversationId}`);
        const data = await response.json();
        
        if (response.ok) {
            currentConversationId = conversationId;
            conversationTitle.textContent = data.title || 'New Conversation';
            
            // Update RAG toggle based on conversation setting
            const ragToggle = document.getElementById('ragToggle');
            if (ragToggle) {
                ragToggle.checked = data.use_rag !== false;
                currentRagEnabled = ragToggle.checked;
            }
            
            // Show RAG toggle container
            const ragToggleContainer = document.getElementById('ragToggleContainer');
            if (ragToggleContainer) {
                ragToggleContainer.style.display = 'flex';
            }
            
            // Clear current context sources
            currentContextSources = [];
            
            // Clear chat messages
            chatMessages.innerHTML = '';
            
            // Add messages to chat
            if (data.messages && data.messages.length > 0) {
                data.messages.forEach(msg => {
                    const contextUsed = msg.context_used || false;
                    const contextSources = msg.context_sources ? JSON.parse(msg.context_sources) : [];
                    addMessageToChat(msg.role, msg.content, contextUsed, contextSources);
                });
            } else {
                // Show welcome message if no messages
                const welcomeMessage = document.querySelector('.welcome-message');
                if (!welcomeMessage) {
                    chatMessages.innerHTML = `
                        <div class="welcome-message">
                            <h3><i class="fas fa-balance-scale"></i> Legal Conversation</h3>
                            <p>Ask legal questions. The assistant will reference relevant legislation.</p>
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

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    loadConversations();
    setupEventListeners();
    adjustTextareaHeight();
    initializeRAG(); // Initialize RAG features
});

// Add notification styles
const style = document.createElement('style');
style.textContent = `
    .notification {
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        z-index: 10000;
        transform: translateX(100%);
        transition: transform 0.3s ease;
        max-width: 300px;
    }
    
    .notification.show {
        transform: translateX(0);
    }
    
    .notification-success {
        background-color: #4caf50;
    }
    
    .notification-info {
        background-color: #2196f3;
    }
    
    .notification-warning {
        background-color: #ff9800;
    }
    
    .notification-error {
        background-color: #f44336;
    }
    
    .doc-type-badge {
        background: #e3f2fd;
        color: #1976d2;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.7rem;
        margin-left: 10px;
    }
    
    .no-results, .no-sources {
        text-align: center;
        padding: 40px 20px;
        color: var(--text-light);
    }
    
    .no-results i, .no-sources i {
        font-size: 3rem;
        margin-bottom: 10px;
        color: #ddd;
    }
`;
document.head.appendChild(style);