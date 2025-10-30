"""
Production-Ready WhatsApp Chat Analyzer API
- Environment-based configuration
- Proper error handling and logging
- Rate limiting and security
- Health checks and monitoring
- Gunicorn-ready
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import json
import re
import os
import logging
from logging.handlers import RotatingFileHandler
from collections import defaultdict
from datetime import datetime
import statistics
from typing import List, Dict
import time
import random
from groq import Groq
from functools import wraps

# ===== CONFIGURATION =====
class Config:
    """Application configuration"""
    # Environment
    ENV = os.getenv('FLASK_ENV', 'production')
    DEBUG = ENV == 'development'
    
    # Security
    SECRET_KEY = os.getenv('SECRET_KEY', os.urandom(24).hex())
    
    # API Keys
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    
    # Rate limiting
    RATELIMIT_ENABLED = os.getenv('RATELIMIT_ENABLED', 'true').lower() == 'true'
    RATELIMIT_DEFAULT = os.getenv('RATELIMIT_DEFAULT', '10 per hour')
    RATELIMIT_STORAGE_URL = os.getenv('REDIS_URL', 'memory://')
    
    # CORS
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'app.log')
    
    # Request limits
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 50 * 1024 * 1024))  # 50MB
    MAX_MESSAGES = int(os.getenv('MAX_MESSAGES', 100000))
    
    # Analysis limits
    ANALYSIS_TIMEOUT = int(os.getenv('ANALYSIS_TIMEOUT', 300))  # 5 minutes


# ===== APP INITIALIZATION =====
app = Flask(__name__)
app.config.from_object(Config)

# CORS setup
CORS(app, origins=Config.CORS_ORIGINS)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=Config.RATELIMIT_STORAGE_URL,
    default_limits=[Config.RATELIMIT_DEFAULT] if Config.RATELIMIT_ENABLED else []
)

# ===== LOGGING SETUP =====
def setup_logging():
    """Configure application logging"""
    log_level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler with rotation
    if not Config.DEBUG:
        file_handler = RotatingFileHandler(
            Config.LOG_FILE,
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        app.logger.addHandler(file_handler)
    
    app.logger.addHandler(console_handler)
    app.logger.setLevel(log_level)
    
    # Suppress werkzeug logs in production
    if not Config.DEBUG:
        logging.getLogger('werkzeug').setLevel(logging.WARNING)

setup_logging()


# ===== MIDDLEWARE & DECORATORS =====
def require_api_key(f):
    """Decorator to validate API key in request"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not Config.GROQ_API_KEY:
            data = request.get_json(silent=True) or {}
            if 'api_key' not in data:
                app.logger.warning("API request without key and no default configured")
                return jsonify({
                    'error': 'API key required',
                    'message': 'Please provide api_key in request body or set GROQ_API_KEY environment variable'
                }), 401
        return f(*args, **kwargs)
    return decorated_function


def validate_chat_data(f):
    """Decorator to validate chat data structure"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            return jsonify({
                'error': 'Invalid content type',
                'message': 'Content-Type must be application/json'
            }), 400
        
        data = request.get_json()
        
        if 'chat_data' not in data:
            return jsonify({
                'error': 'Missing chat_data',
                'message': 'Request must include chat_data field'
            }), 400
        
        chat_data = data['chat_data']
        
        # Validate required fields
        required_fields = ['messages', 'participants']
        for field in required_fields:
            if field not in chat_data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'message': f'chat_data must include {field}'
                }), 400
        
        # Validate message count
        if len(chat_data['messages']) > Config.MAX_MESSAGES:
            return jsonify({
                'error': 'Too many messages',
                'message': f'Maximum {Config.MAX_MESSAGES} messages allowed'
            }), 413
        
        return f(*args, **kwargs)
    return decorated_function


@app.before_request
def log_request():
    """Log incoming requests"""
    if not request.path.startswith('/health'):
        app.logger.info(f"{request.method} {request.path} from {request.remote_addr}")


@app.after_request
def after_request(response):
    """Add security headers"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle payload too large errors"""
    return jsonify({
        'error': 'Payload too large',
        'message': f'Maximum request size is {Config.MAX_CONTENT_LENGTH / 1024 / 1024}MB'
    }), 413


@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit errors"""
    app.logger.warning(f"Rate limit exceeded: {request.remote_addr}")
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': str(e.description)
    }), 429


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    app.logger.error(f"Internal error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


# ===== ANALYZER CLASS =====
class EnhancedWhatsAppAnalyzer:
    def __init__(self, chat_data: dict, groq_api_key: str):
        """Initialize analyzer with chat data and Groq API"""
        self.data = chat_data
        self.messages = self.data['messages']
        
        excluded = ['Meta AI', '+91']
        self.participants = [p for p in self.data['participants'] 
                           if not any(ex in p for ex in excluded)]
        self.stats = self.data.get('stats', {})
        self.client = Groq(api_key=groq_api_key)
        
        self.messages = [m for m in self.messages if m['sender'] in self.participants]
        
        app.logger.info(f"Loaded {len(self.messages):,} messages from {len(self.participants)} participants")
    
    def sample_messages(self, n: int = 500, strategy: str = 'smart') -> List[Dict]:
        """Intelligent sampling for maximum diversity"""
        if n >= len(self.messages):
            return self.messages
        
        if strategy == 'random':
            return random.sample(self.messages, n)
        elif strategy == 'distributed':
            step = len(self.messages) // n
            return [self.messages[i] for i in range(0, len(self.messages), step)][:n]
        elif strategy == 'smart':
            distributed_n = int(n * 0.4)
            long_n = int(n * 0.4)
            random_n = n - distributed_n - long_n
            
            step = len(self.messages) // distributed_n if distributed_n > 0 else 1
            distributed = [self.messages[i] for i in range(0, len(self.messages), step)][:distributed_n]
            
            sorted_by_length = sorted(self.messages, key=lambda x: len(x['message']), reverse=True)
            long_messages = sorted_by_length[:long_n]
            
            remaining = [m for m in self.messages if m not in distributed and m not in long_messages]
            random_sample = random.sample(remaining, min(random_n, len(remaining)))
            
            combined = distributed + long_messages + random_sample
            random.shuffle(combined)
            return combined
        
        return self.messages[:n]
    
    def format_messages_for_llm(self, messages: List[Dict], max_msgs: int = 100) -> str:
        """Format messages for LLM context"""
        context = ""
        for msg in messages[:max_msgs]:
            sender = msg['sender']
            text = msg['message'][:200]
            date = msg.get('date', '')
            context += f"[{date}] {sender}: {text}\n"
        return context
    
    def detect_emoji_usage(self) -> Dict[str, int]:
        """Count emoji usage per person"""
        emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
            r'\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+'
        )
        emoji_counts = defaultdict(int)
        
        for msg in self.messages:
            emojis = emoji_pattern.findall(msg['message'])
            emoji_counts[msg['sender']] += len(emojis)
        
        return dict(emoji_counts)
    
    def extract_basic_stats(self) -> Dict:
        """Quick statistical baseline"""
        msg_lengths = defaultdict(list)
        response_times = defaultdict(list)
        
        for i, msg in enumerate(self.messages):
            sender = msg['sender']
            msg_lengths[sender].append(len(msg['message']))
            
            if i > 0 and self.messages[i-1]['sender'] != sender:
                try:
                    curr_time = datetime.strptime(
                        f"{msg['date']} {msg['time']}", "%d/%m/%y %H:%M"
                    )
                    prev_time = datetime.strptime(
                        f"{self.messages[i-1]['date']} {self.messages[i-1]['time']}", 
                        "%d/%m/%y %H:%M"
                    )
                    delta = (curr_time - prev_time).total_seconds() / 60
                    if delta < 60:
                        response_times[sender].append(delta)
                except:
                    pass
        
        return {
            'avg_message_length': {
                p: round(statistics.mean(msg_lengths[p]), 1) 
                for p in self.participants if msg_lengths[p]
            },
            'avg_response_time_minutes': {
                p: round(statistics.mean(response_times[p]), 1)
                for p in self.participants if response_times[p]
            }
        }
    
    def _call_groq_api(self, prompt: str, system_msg: str, max_tokens: int = 4000, temp: float = 0.4) -> Dict:
        """Centralized Groq API call with error handling"""
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=temp,
                max_tokens=max_tokens
            )
            
            result_text = response.choices[0].message.content.strip()
            result_text = self._clean_json_response(result_text)
            return json.loads(result_text)
        
        except json.JSONDecodeError as e:
            app.logger.error(f"JSON decode error: {e}")
            raise ValueError(f"Invalid JSON response from API: {str(e)}")
        except Exception as e:
            app.logger.error(f"Groq API error: {e}")
            raise
    
    def _clean_json_response(self, text: str) -> str:
        """Clean markdown from JSON"""
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
        return text
    
    def ai_enhanced_personality_analysis(self, sample_size: int = 600) -> Dict:
        """Enhanced personality + D&D alignment analysis"""
        sampled = self.sample_messages(sample_size, 'smart')
        context = self.format_messages_for_llm(sampled, max_msgs=120)
        
        prompt = f"""Analyze this multilingual WhatsApp chat (English/Hinglish/mixed languages).

PARTICIPANTS: {', '.join(self.participants)}

MESSAGES:
{context}

Provide comprehensive analysis:

1. PERSONALITY PROFILES: For each participant:
   - Communication style (formal/casual/energetic/calm/aggressive/passive)
   - Emotional tone (supportive/sarcastic/neutral/humorous/dramatic)
   - Key traits (3-4 specific traits)

2. ENHANCED ROLES (score 0-100 with specific examples):
   - Therapist: Most emotionally supportive
   - Hype Man: Most encouraging/energetic
   - Comedian: Funniest/most entertaining
   - Intellectual: Most thoughtful/insightful
   - Peacemaker: Best at diffusing tension
   - Drama Queen: Most dramatic/attention-seeking
   - Meme Lord: Best at memes/internet culture
   - Ghost: Lurks then appears with perfect timing
   - Chaos Agent: Creates fun chaos/unpredictability
   - Voice of Reason: Most logical/grounded
   - Oversharer: Shares the most personal details
   - Conspiracy Theorist: Most random theories/wild takes

3. D&D ALIGNMENT (be specific and nuanced):
   Assign alignment based on behavior patterns.

4. COMMUNICATION PATTERNS:
   - Who initiates serious conversations?
   - Who keeps things light?
   - Who maintains group connection?

Respond with ONLY valid JSON with all required fields."""
        
        system_msg = "You are a multilingual chat analysis expert. Understand Hinglish and code-switching. Respond with valid JSON only."
        
        return self._call_groq_api(prompt, system_msg, max_tokens=4000, temp=0.4)
    
    def ai_golden_moments_analysis(self, sample_size: int = 700) -> Dict:
        """Expanded golden moments with 8-10 categories"""
        sampled = self.sample_messages(sample_size, 'smart')
        context = self.format_messages_for_llm(sampled, max_msgs=150)
        
        prompt = f"""Analyze memorable moments in this multilingual chat.

PARTICIPANTS: {', '.join(self.participants)}

MESSAGES:
{context}

Find the BEST examples for 10 categories: Most Wholesome, Most Savage, Most Random, Most Inspirational, Most Cringe, Most Relatable, Most Unhinged, Plot Twist, Mic Drop, Big Brain.

Each must be actually impactful with clear context. ONLY include categories with STRONG examples.

Respond with ONLY valid JSON."""
        
        system_msg = "You are a content curator. Find genuinely memorable moments. Skip categories without strong examples. Respond with JSON only."
        
        result = self._call_groq_api(prompt, system_msg, max_tokens=3500, temp=0.5)
        
        # Filter low-quality moments
        if 'golden_moments' in result:
            result['golden_moments'] = [
                m for m in result['golden_moments']
                if m.get('impact_score', 0) >= 70 and len(m.get('message', '')) > 20
            ]
        
        return result
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate full analysis"""
        app.logger.info("Starting comprehensive analysis")
        start_time = time.time()
        
        report = {
            'metadata': {
                'total_messages': len(self.messages),
                'participants': self.participants,
                'date_range': self.data.get('dateRange', {}),
                'analysis_timestamp': datetime.utcnow().isoformat() + 'Z',
                'model': 'llama-3.3-70b-versatile'
            }
        }
        
        try:
            # Statistical baseline
            app.logger.info("Extracting basic stats")
            report['basic_stats'] = self.extract_basic_stats()
            report['emoji_usage'] = self.detect_emoji_usage()
            
            # AI analyses
            app.logger.info("Running personality analysis")
            personality_data = self.ai_enhanced_personality_analysis(600)
            report['personalities'] = personality_data.get('personalities', [])
            report['roles'] = personality_data.get('roles', {})
            report['alignments'] = personality_data.get('alignments', [])
            report['communication_patterns'] = personality_data.get('communication_patterns', {})
            time.sleep(0.5)
            
            app.logger.info("Running golden moments analysis")
            moments_data = self.ai_golden_moments_analysis(700)
            report['golden_moments'] = moments_data.get('golden_moments', [])
            
            elapsed = time.time() - start_time
            report['metadata']['analysis_time_seconds'] = round(elapsed, 2)
            
            app.logger.info(f"Analysis complete in {elapsed:.1f}s")
            
            return report
            
        except Exception as e:
            app.logger.error(f"Analysis failed: {str(e)}")
            raise


# ===== API ROUTES =====

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API documentation"""
    return jsonify({
        'service': 'WhatsApp Chat Analyzer API',
        'version': '2.0',
        'status': 'operational',
        'endpoints': {
            'health': 'GET /health - Health check',
            'analyze': 'POST /analyze - Full comprehensive analysis',
            'quick_analyze': 'POST /analyze/quick - Quick analysis'
        },
        'documentation': 'See README for payload format'
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Detailed health check endpoint"""
    groq_configured = bool(Config.GROQ_API_KEY)
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'service': 'WhatsApp Chat Analyzer',
        'version': '2.0',
        'environment': Config.ENV,
        'groq_api_configured': groq_configured,
        'rate_limiting_enabled': Config.RATELIMIT_ENABLED
    })


@app.route('/analyze', methods=['POST'])
@limiter.limit("5 per hour")
@require_api_key
@validate_chat_data
def analyze_chat():
    """
    Full comprehensive analysis endpoint
    
    Expected JSON payload:
    {
        "chat_data": {
            "messages": [...],
            "participants": [...],
            "dateRange": {...}
        },
        "api_key": "..."  // Optional if GROQ_API_KEY env var set
    }
    """
    try:
        data = request.get_json()
        chat_data = data['chat_data']
        api_key = data.get('api_key', Config.GROQ_API_KEY)
        
        app.logger.info(f"Starting analysis: {len(chat_data['messages'])} messages, {len(chat_data['participants'])} participants")
        
        analyzer = EnhancedWhatsAppAnalyzer(chat_data, api_key)
        report = analyzer.generate_comprehensive_report()
        
        return jsonify({
            'success': True,
            'report': report
        })
    
    except ValueError as e:
        app.logger.error(f"Validation error: {str(e)}")
        return jsonify({
            'error': 'Validation error',
            'message': str(e)
        }), 400
    
    except Exception as e:
        app.logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Analysis failed',
            'message': str(e) if Config.DEBUG else 'An error occurred during analysis'
        }), 500


@app.route('/analyze/quick', methods=['POST'])
@limiter.limit("10 per hour")
@require_api_key
@validate_chat_data
def quick_analyze():
    """
    Quick analysis endpoint (fewer API calls, faster)
    Same payload as /analyze but with reduced sample sizes
    """
    try:
        data = request.get_json()
        chat_data = data['chat_data']
        api_key = data.get('api_key', Config.GROQ_API_KEY)
        
        app.logger.info(f"Starting quick analysis: {len(chat_data['messages'])} messages")
        
        analyzer = EnhancedWhatsAppAnalyzer(chat_data, api_key)
        
        report = {
            'metadata': {
                'total_messages': len(analyzer.messages),
                'participants': analyzer.participants,
                'analysis_type': 'quick',
                'analysis_timestamp': datetime.utcnow().isoformat() + 'Z'
            }
        }
        
        # Only run 2 core analyses
        report['basic_stats'] = analyzer.extract_basic_stats()
        
        personality_data = analyzer.ai_enhanced_personality_analysis(400)
        report['personalities'] = personality_data.get('personalities', [])
        report['roles'] = personality_data.get('roles', {})
        report['alignments'] = personality_data.get('alignments', [])
        
        moments_data = analyzer.ai_golden_moments_analysis(400)
        report['golden_moments'] = moments_data.get('golden_moments', [])
        
        return jsonify({
            'success': True,
            'report': report
        })
    
    except Exception as e:
        app.logger.error(f"Quick analysis failed: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Quick analysis failed',
            'message': str(e) if Config.DEBUG else 'An error occurred during analysis'
        }), 500


# ===== MAIN =====
if __name__ == '__main__':
    if Config.DEBUG:
        app.logger.warning("Running in DEBUG mode - not suitable for production!")
        app.run(host='0.0.0.0', port=5050, debug=True)
    else:
        app.logger.info("Starting in production mode")
        app.logger.info("Use gunicorn for production: gunicorn -w 4 -b 0.0.0.0:5050 app:app")
        app.run(host='0.0.0.0', port=5050)