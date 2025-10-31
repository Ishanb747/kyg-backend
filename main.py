"""
Enhanced WhatsApp Chat Analyzer - Flask API
Improvements:
- Nuanced D&D alignments based on behavior patterns
- 8-10 golden moment categories with filtering
- Expanded spicy roles (Drama Queen, Meme Lord, Ghost, etc.)
- Flask API with JSON upload and optional API key
- Environment variable configuration for deployment
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import re
from collections import defaultdict, Counter
from datetime import datetime
import statistics
from typing import List, Dict, Tuple
import time
import random
from groq import Groq
import os

app = Flask(__name__)
CORS(app)

# Get API key from environment variable
DEFAULT_GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')


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
        
        print(f"✓ Loaded {len(self.messages):,} messages from {len(self.participants)} participants")
    
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
   Assign alignment based on:
   - LAWFUL: Structured, rule-following, organized, planners
   - NEUTRAL: Balanced, adaptable, situation-dependent
   - CHAOTIC: Spontaneous, unpredictable, rule-breaking, random
   
   - GOOD: Supportive, kind, helps others, uplifts group
   - NEUTRAL: Balanced morality, self-focused but not harmful
   - EVIL: Teases/roasts others, sarcastic, playfully mean (not actually evil)
   
   Examples:
   - Lawful Good: Organized helper, plans group activities
   - Neutral Good: Supportive but spontaneous
   - Chaotic Good: Random acts of kindness, unpredictable support
   - Lawful Neutral: By-the-book, matter-of-fact
   - True Neutral: Goes with the flow, balanced
   - Chaotic Neutral: Pure chaos, no pattern
   - Lawful Evil: Organized roaster, calculated sarcasm
   - Neutral Evil: Opportunistic teaser
   - Chaotic Evil: Random savage roasts, unpredictable meanness

4. COMMUNICATION PATTERNS:
   - Who initiates serious conversations?
   - Who keeps things light?
   - Who maintains group connection?

Respond with ONLY valid JSON:
{{
  "personalities": [
    {{
      "name": "PersonName",
      "style": "description",
      "tone": "description",
      "traits": ["trait1", "trait2", "trait3", "trait4"]
    }}
  ],
  "roles": {{
    "therapist": {{"name": "Name", "score": 85, "reason": "specific examples"}},
    "hype_man": {{"name": "Name", "score": 90, "reason": "specific examples"}},
    "comedian": {{"name": "Name", "score": 88, "reason": "specific examples"}},
    "intellectual": {{"name": "Name", "score": 82, "reason": "specific examples"}},
    "peacemaker": {{"name": "Name", "score": 75, "reason": "specific examples"}},
    "drama_queen": {{"name": "Name", "score": 80, "reason": "specific examples"}},
    "meme_lord": {{"name": "Name", "score": 85, "reason": "specific examples"}},
    "ghost": {{"name": "Name", "score": 70, "reason": "specific examples"}},
    "chaos_agent": {{"name": "Name", "score": 88, "reason": "specific examples"}},
    "voice_of_reason": {{"name": "Name", "score": 83, "reason": "specific examples"}},
    "oversharer": {{"name": "Name", "score": 77, "reason": "specific examples"}},
    "conspiracy_theorist": {{"name": "Name", "score": 72, "reason": "specific examples"}}
  }},
  "alignments": [
    {{
      "name": "Person1",
      "alignment": "Chaotic Good",
      "reason": "specific behavioral evidence"
    }}
  ],
  "communication_patterns": {{
    "serious_initiator": "Name",
    "lightness_keeper": "Name",
    "connection_maintainer": "Name"
  }}
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a multilingual chat analysis expert. Understand Hinglish and code-switching. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=4000
            )
            
            result_text = response.choices[0].message.content.strip()
            result_text = self._clean_json_response(result_text)
            return json.loads(result_text)
        
        except Exception as e:
            print(f"⚠️  Groq API error (personality): {e}")
            return {}
    
    def ai_golden_moments_analysis(self, sample_size: int = 700) -> Dict:
        """Expanded golden moments with 8-10 categories"""
        sampled = self.sample_messages(sample_size, 'smart')
        context = self.format_messages_for_llm(sampled, max_msgs=150)
        
        prompt = f"""Analyze memorable moments in this multilingual chat.

PARTICIPANTS: {', '.join(self.participants)}

MESSAGES:
{context}

Find the BEST examples for each category. Each must be:
- Actually impactful/meaningful/funny (not generic)
- Have clear context
- Representative of the category
- Include actual message snippet (30-100 chars)

GOLDEN MOMENT CATEGORIES:
1. Most Wholesome: Genuinely heartwarming message
2. Most Savage: Wittiest roast/comeback
3. Most Random: Hilariously out-of-context
4. Most Inspirational: Motivational/uplifting
5. Most Cringe: Awkward but funny
6. Most Relatable: Universal "same bro" moment
7. Most Unhinged: Absolutely wild/crazy statement
8. Plot Twist: Unexpected conversation turn
9. Mic Drop: Perfect conversation ender
10. Big Brain: Brilliant insight/solution

ONLY include categories where you found CLEAR, STRONG examples. Skip if no good match.

Respond with ONLY valid JSON:
{{
  "golden_moments": [
    {{
      "category": "Most Savage",
      "sender": "Name",
      "message": "message snippet (30-100 chars)",
      "context": "why this is perfect for category",
      "impact_score": 95
    }}
  ]
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a content curator. Find genuinely memorable moments, not generic ones. Skip categories without strong examples. Respond with JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=3500
            )
            
            result_text = response.choices[0].message.content.strip()
            result_text = self._clean_json_response(result_text)
            parsed = json.loads(result_text)
            
            # Filter out low-quality moments
            if 'golden_moments' in parsed:
                parsed['golden_moments'] = [
                    m for m in parsed['golden_moments']
                    if m.get('impact_score', 0) >= 70 and len(m.get('message', '')) > 20
                ]
            
            return parsed
        
        except Exception as e:
            print(f"⚠️  Groq API error (golden moments): {e}")
            return {}
    
    def ai_content_analysis(self, sample_size: int = 700) -> Dict:
        """Topics, humor, and language patterns"""
        sampled = self.sample_messages(sample_size, 'smart')
        context = self.format_messages_for_llm(sampled, max_msgs=150)
        
        prompt = f"""Analyze content themes in this multilingual chat.

PARTICIPANTS: {', '.join(self.participants)}

MESSAGES:
{context}

Analyze:

1. TOP TOPICS (top 10):
   - Topic name
   - Frequency (high/medium/low)
   - Key participants
   - Sample keywords (including Hinglish)

2. HUMOR ANALYSIS:
   - Top 5 funniest messages with context
   - Top 5 sarcastic moments
   - Inside jokes (recurring phrases)

3. LANGUAGE PATTERNS:
   - Unique slang/Hinglish terms
   - Code-switching patterns

Respond with ONLY valid JSON:
{{
  "topics": [
    {{
      "topic": "Topic Name",
      "frequency": "high",
      "participants": ["Name1"],
      "keywords": ["word1", "word2"]
    }}
  ],
  "humor": {{
    "funniest": [
      {{"sender": "Name", "message": "snippet", "why": "explanation"}}
    ],
    "sarcasm": [
      {{"sender": "Name", "message": "snippet", "context": "why sarcastic"}}
    ],
    "inside_jokes": [
      {{"phrase": "recurring phrase", "usage": "how used", "frequency": "medium"}}
    ]
  }},
  "language_patterns": {{
    "unique_slang": ["term1", "term2"],
    "code_switching": "description"
  }}
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a multilingual content analyst. Respond with JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=3500
            )
            
            result_text = response.choices[0].message.content.strip()
            result_text = self._clean_json_response(result_text)
            return json.loads(result_text)
        
        except Exception as e:
            print(f"⚠️  Groq API error (content): {e}")
            return {}
    
    def ai_relationship_dynamics(self, sample_size: int = 600) -> Dict:
        """Relationship analysis"""
        sampled = self.sample_messages(sample_size, 'smart')
        context = self.format_messages_for_llm(sampled, max_msgs=120)
        
        prompt = f"""Analyze relationships in this chat.

PARTICIPANTS: {', '.join(self.participants)}

MESSAGES:
{context}

Analyze:

1. CLOSEST PAIRS: Top 3-5 pairs with:
   - Bond type
   - Dynamic description
   - Evidence

2. GROUP DYNAMICS:
   - Energy Matcher: Best at matching others' vibe
   - Conflict Resolver: Calms arguments
   - Topic Starter: Initiates discussions
   - Silent Observer: Selective but meaningful participation

3. INTERACTION PATTERNS:
   - Response patterns
   - Leadership style

Respond with ONLY valid JSON:
{{
  "closest_pairs": [
    {{
      "pair": ["Person1", "Person2"],
      "bond_type": "type",
      "dynamic": "description",
      "evidence": "examples"
    }}
  ],
  "group_roles": {{
    "energy_matcher": {{"name": "Name", "score": 85, "reason": "why"}},
    "conflict_resolver": {{"name": "Name", "score": 80, "reason": "why"}},
    "topic_starter": {{"name": "Name", "score": 90, "reason": "why"}},
    "silent_observer": {{"name": "Name", "score": 75, "reason": "why"}}
  }},
  "dynamics": {{
    "leadership": "description",
    "subgroups": "description or null"
  }}
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a social dynamics expert. Respond with JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=3000
            )
            
            result_text = response.choices[0].message.content.strip()
            result_text = self._clean_json_response(result_text)
            return json.loads(result_text)
        
        except Exception as e:
            print(f"⚠️  Groq API error (relationships): {e}")
            return {}
    
    def ai_sentiment_timeline(self, sample_size: int = 500) -> Dict:
        """Sentiment analysis over time"""
        sampled = self.sample_messages(sample_size, 'distributed')
        
        period_size = len(sampled) // 8
        periods = [sampled[i:i+period_size] for i in range(0, len(sampled), period_size)]
        
        period_summaries = []
        for idx, period in enumerate(periods[:8]):
            if not period:
                continue
            
            date_range = f"{period[0]['date']} to {period[-1]['date']}"
            context = self.format_messages_for_llm(period, max_msgs=30)
            
            period_summaries.append({
                'period': idx + 1,
                'date_range': date_range,
                'sample': context
            })
        
        periods_text = "\n\n".join([
            f"PERIOD {p['period']} ({p['date_range']}):\n{p['sample']}"
            for p in period_summaries
        ])
        
        prompt = f"""Analyze sentiment evolution.

PARTICIPANTS: {', '.join(self.participants)}

{periods_text}

For each period:
1. Mood (positive/neutral/negative/mixed)
2. Energy (high/medium/low)
3. Key themes
4. Shifts from previous

Respond with ONLY valid JSON:
{{
  "timeline": [
    {{
      "period": 1,
      "date_range": "range",
      "mood": "positive",
      "energy": "high",
      "themes": ["theme1"],
      "shift": "change description"
    }}
  ],
  "overall_trend": "improving/stable/declining/fluctuating",
  "healthiest_period": 3,
  "most_active_period": 5
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a sentiment analyst. Respond with JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2500
            )
            
            result_text = response.choices[0].message.content.strip()
            result_text = self._clean_json_response(result_text)
            return json.loads(result_text)
        
        except Exception as e:
            print(f"⚠️  Groq API error (sentiment): {e}")
            return {}
    
    def _clean_json_response(self, text: str) -> str:
        """Clean markdown from JSON"""
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
        return text
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate full analysis"""
        print("\n" + "="*70)
        print("ENHANCED AI-POWERED CHAT ANALYSIS")
        print("="*70 + "\n")
        start_time = time.time()
        
        report = {
            'metadata': {
                'total_messages': len(self.messages),
                'participants': self.participants,
                'date_range': self.data.get('dateRange', {}),
                'analysis_timestamp': datetime.now().isoformat(),
                'model': 'llama-3.3-70b-versatile'
            }
        }
        
        # Statistical baseline
        print("📊 Statistical Baseline...")
        report['basic_stats'] = self.extract_basic_stats()
        report['emoji_usage'] = self.detect_emoji_usage()
        
        # AI analyses
        print("🤖 AI Analysis [1/5]: Personalities & D&D Alignments...")
        personality_data = self.ai_enhanced_personality_analysis(600)
        report['personalities'] = personality_data.get('personalities', [])
        report['roles'] = personality_data.get('roles', {})
        report['alignments'] = personality_data.get('alignments', [])
        report['communication_patterns'] = personality_data.get('communication_patterns', {})
        time.sleep(0.5)
        
        print("🤖 AI Analysis [2/5]: Golden Moments...")
        moments_data = self.ai_golden_moments_analysis(700)
        report['golden_moments'] = moments_data.get('golden_moments', [])
        time.sleep(0.5)
        
        print("🤖 AI Analysis [3/5]: Content & Humor...")
        content_data = self.ai_content_analysis(700)
        report['topics'] = content_data.get('topics', [])
        report['humor'] = content_data.get('humor', {})
        report['language_patterns'] = content_data.get('language_patterns', {})
        time.sleep(0.5)
        
        print("🤖 AI Analysis [4/5]: Relationships...")
        relationship_data = self.ai_relationship_dynamics(600)
        report['closest_pairs'] = relationship_data.get('closest_pairs', [])
        report['group_roles'] = relationship_data.get('group_roles', {})
        report['group_dynamics'] = relationship_data.get('dynamics', {})
        time.sleep(0.5)
        
        print("🤖 AI Analysis [5/5]: Sentiment Timeline...")
        sentiment_data = self.ai_sentiment_timeline(500)
        report['sentiment_timeline'] = sentiment_data.get('timeline', [])
        report['sentiment_trend'] = sentiment_data.get('overall_trend', 'unknown')
        
        elapsed = time.time() - start_time
        report['metadata']['analysis_time_seconds'] = round(elapsed, 2)
        
        print(f"\n✅ Analysis complete in {elapsed:.1f}s\n")
        
        return report


# ===== FLASK API =====

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'WhatsApp Chat Analyzer',
        'version': '2.0',
        'api_key_configured': bool(DEFAULT_GROQ_API_KEY)
    })


@app.route('/analyze', methods=['POST'])
def analyze_chat():
    """
    Main analysis endpoint
    
    Expected JSON payload:
    {
        "chat_data": { ... },  // Required: WhatsApp chat JSON
        "api_key": "..."       // Optional: Groq API key (uses env var if not provided)
    }
    """
    try:
        # Parse request
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type must be application/json'
            }), 400
        
        data = request.get_json()
        
        # Validate chat_data
        if 'chat_data' not in data:
            return jsonify({
                'error': 'Missing required field: chat_data'
            }), 400
        
        chat_data = data['chat_data']
        
        # Validate structure
        required_fields = ['messages', 'participants']
        for field in required_fields:
            if field not in chat_data:
                return jsonify({
                    'error': f'chat_data missing required field: {field}'
                }), 400
        
        # Get API key (priority: request > env var)
        api_key = data.get('api_key', DEFAULT_GROQ_API_KEY)
        
        if not api_key:
            return jsonify({
                'error': 'No API key provided. Set GROQ_API_KEY environment variable or include api_key in request'
            }), 400
        
        # Run analysis
        print(f"\n{'='*70}")
        print(f"NEW ANALYSIS REQUEST")
        print(f"Messages: {len(chat_data['messages'])}")
        print(f"Participants: {len(chat_data['participants'])}")
        print(f"Using {'custom' if 'api_key' in data else 'environment'} API key")
        print(f"{'='*70}")
        
        analyzer = EnhancedWhatsAppAnalyzer(chat_data, api_key)
        report = analyzer.generate_comprehensive_report()
        
        return jsonify({
            'success': True,
            'report': report
        })
    
    except json.JSONDecodeError as e:
        return jsonify({
            'error': 'Invalid JSON format',
            'details': str(e)
        }), 400
    
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return jsonify({
            'error': 'Analysis failed',
            'details': str(e)
        }), 500


@app.route('/analyze/quick', methods=['POST'])
def quick_analyze():
    """
    Quick analysis endpoint (fewer API calls, faster)
    
    Same payload as /analyze but runs with reduced sample sizes
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        
        if 'chat_data' not in data:
            return jsonify({'error': 'Missing required field: chat_data'}), 400
        
        chat_data = data['chat_data']
        api_key = data.get('api_key', DEFAULT_GROQ_API_KEY)
        
        if not api_key:
            return jsonify({
                'error': 'No API key provided. Set GROQ_API_KEY environment variable or include api_key in request'
            }), 400
        
        # Quick analysis with smaller samples
        analyzer = EnhancedWhatsAppAnalyzer(chat_data, api_key)
        
        report = {
            'metadata': {
                'total_messages': len(analyzer.messages),
                'participants': analyzer.participants,
                'analysis_type': 'quick'
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
        return jsonify({
            'error': 'Quick analysis failed',
            'details': str(e)
        }), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Enhanced WhatsApp Analyzer API Starting...")
    print("="*70)
    print("\nEnvironment Configuration:")
    print(f"  GROQ_API_KEY: {'✓ Set' if DEFAULT_GROQ_API_KEY else '✗ Not set (must be provided in requests)'}")
    print("\nEndpoints:")
    print("  GET  /health          - Health check")
    print("  POST /analyze         - Full comprehensive analysis")
    print("  POST /analyze/quick   - Quick analysis (2-3 API calls)")
    print("\nPayload format:")
    print("""
  {
    "chat_data": {
      "messages": [...],
      "participants": [...],
      "dateRange": {...}
    },
    "api_key": "optional-groq-key"  // Optional if GROQ_API_KEY env var is set
  }
    """)
    print("="*70 + "\n")
    
    # Run on all interfaces, port 5050
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=False)