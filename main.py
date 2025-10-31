"""
Enhanced Token-Optimized WhatsApp Chat Analyzer v3.1
- Improved multi-strategy sampling (temporal + hotspots + quality)
- Better conversation thread detection
- More efficient token usage
- Context messages for golden moments
- Extended meme analysis
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import re
from collections import defaultdict, Counter
from datetime import datetime
import statistics
from typing import List, Dict
import time
import random
from groq import Groq
import os

app = Flask(__name__)

# Enhanced CORS configuration
CORS(app, origins=[
    "http://localhost:5173",
    "https://kyg-frontend.onrender.com"
])

# Additional CORS headers for preflight
@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin in ["http://localhost:5173", "https://kyg-frontend.onrender.com"]:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"  # prevent caching issues
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

DEFAULT_GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')


class OptimizedWhatsAppAnalyzer:
    def __init__(self, chat_data: dict, groq_api_key: str):
        self.data = chat_data
        self.all_messages = self.data['messages']  # Keep original
        
        excluded = ['Meta AI', '+91']
        self.participants = [p for p in self.data['participants'] 
                           if not any(ex in p for ex in excluded)]
        self.client = Groq(api_key=groq_api_key)
        
        # Filter messages but keep originals indexed
        self.messages = [m for m in self.all_messages if m['sender'] in self.participants]
        
        # Create index for context extraction (before preprocessing)
        self.message_index = {i: msg for i, msg in enumerate(self.messages)}
        
        # Preprocess for quality
        self.processed_messages = self._preprocess_messages()
        
        print(f"✓ Loaded {len(self.processed_messages):,} quality messages from {len(self.participants)} participants")
    
    def _preprocess_messages(self) -> List[Dict]:
        """Remove noise and low-value messages"""
        cleaned = []
        seen_texts = set()
        
        for msg in self.messages:
            text = msg['message'].strip()
            
            # Skip noise
            if len(text) < 3:
                continue
            if text in ['👍', '👎', '❤️', '😂', '🙏']:
                continue
            if text.lower() in ['ok', 'okay', 'k', 'hmm', 'yes', 'no', 'haan', 'nahi']:
                continue
            if text == '<Media omitted>':
                continue
            
            # Skip duplicates
            msg_key = f"{msg['sender']}:{text[:50]}"
            if msg_key in seen_texts:
                continue
            seen_texts.add(msg_key)
            
            cleaned.append(msg)
        
        return cleaned
    
    def get_context_messages(self, message_text: str, sender: str, before: int = 2, after: int = 1) -> List[Dict]:
        """Extract context messages around a target message"""
        try:
            # Find the message in original messages
            target_idx = None
            for idx, msg in self.message_index.items():
                if msg['sender'] == sender and message_text[:30].lower() in msg['message'].lower():
                    target_idx = idx
                    break
            
            if target_idx is None:
                return []
            
            context = []
            # Get before messages
            for i in range(max(0, target_idx - before), target_idx):
                if i in self.message_index:
                    msg = self.message_index[i]
                    if msg['message'] != '<Media omitted>' and len(msg['message']) > 2:
                        context.append(msg)
            
            # Add target
            context.append(self.message_index[target_idx])
            
            # Get after messages
            for i in range(target_idx + 1, min(len(self.message_index), target_idx + after + 1)):
                if i in self.message_index:
                    msg = self.message_index[i]
                    if msg['message'] != '<Media omitted>' and len(msg['message']) > 2:
                        context.append(msg)
            
            return context[-4:]  # Max 4 messages
        except Exception as e:
            print(f"⚠️ Context extraction error: {e}")
            return []
    
    def _temporal_stratified_sample(self, n: int) -> List[Dict]:
        """Sample evenly across chat timeline for temporal coverage"""
        if n >= len(self.processed_messages):
            return self.processed_messages
        
        # Divide into time buckets (5 periods: early, early-mid, mid, mid-late, late)
        num_buckets = 5
        bucket_size = len(self.processed_messages) // num_buckets
        per_bucket = n // num_buckets
        
        samples = []
        for i in range(num_buckets):
            start = i * bucket_size
            end = start + bucket_size if i < num_buckets - 1 else len(self.processed_messages)
            bucket = self.processed_messages[start:end]
            
            if len(bucket) <= per_bucket:
                samples.extend(bucket)
            else:
                # Take evenly spaced messages from bucket
                step = max(1, len(bucket) // per_bucket)
                samples.extend([bucket[j] for j in range(0, len(bucket), step)][:per_bucket])
        
        return samples
    
    def _activity_hotspot_sample(self, n: int) -> List[Dict]:
        """
        Identify periods with high activity (interesting conversations)
        and sample conversation threads from them
        """
        if n >= len(self.processed_messages):
            return self.processed_messages
        
        # Calculate activity density (messages per time window)
        window_size = 20  # messages per window
        hotspots = []
        
        for i in range(0, len(self.processed_messages) - window_size, 5):
            window = self.processed_messages[i:i + window_size]
            
            # Score this window
            score = 0
            unique_senders = len(set(msg['sender'] for msg in window))
            avg_length = sum(len(msg['message']) for msg in window) / len(window)
            
            score += unique_senders * 10  # Conversation diversity
            score += min(avg_length / 5, 20)  # Substantial messages
            
            # Check for engagement indicators
            for msg in window:
                text = msg['message'].lower()
                if '?' in text:
                    score += 5
                if any(word in text for word in ['haha', 'lol', 'omg', 'wtf', '😂', '🤣']):
                    score += 3
            
            hotspots.append((score, i, window))
        
        # Sort by score and take top hotspots
        hotspots.sort(reverse=True, key=lambda x: x[0])
        
        # Sample conversation threads from hotspots
        samples = []
        threads_needed = max(1, n // 3)  # Each thread ~3 messages
        
        for score, start_idx, window in hotspots[:threads_needed]:
            if len(samples) >= n:
                break
            # Take 2-4 consecutive messages from this hotspot
            thread_start = random.randint(0, max(0, len(window) - 4))
            thread_length = random.randint(2, 4)
            thread = window[thread_start:thread_start + thread_length]
            samples.extend(thread)
        
        return samples[:n]
    
    def _quality_based_sample(self, n: int) -> List[Dict]:
        """Sample high-quality individual messages"""
        scored = []
        
        for msg in self.processed_messages:
            score = 0
            text = msg['message']
            text_lower = text.lower()
            
            # Length (but not too long)
            length_score = min(len(text) / 8, 25) - (max(0, len(text) - 200) / 10)
            score += length_score
            
            # Questions and engagement
            if '?' in text:
                score += 15
            if any(word in text_lower for word in ['why', 'how', 'what', 'when', 'where']):
                score += 8
            
            # Emotional/interesting content
            interesting_words = [
                'love', 'hate', 'amazing', 'terrible', 'crazy', 'wtf', 'omg',
                'literally', 'honestly', 'obviously', 'actually', 'seriously'
            ]
            score += sum(3 for word in interesting_words if word in text_lower)
            
            # Reactions and humor
            humor_indicators = ['haha', 'lol', 'lmao', 'rofl', '😂', '🤣', '💀']
            score += sum(4 for indicator in humor_indicators if indicator in text_lower)
            
            scored.append((score, msg))
        
        # Sort and take top
        scored.sort(reverse=True, key=lambda x: x[0])
        return [msg for score, msg in scored[:n]]
    
    def smart_sample(self, n: int = 350) -> List[Dict]:
        """
        Multi-strategy sampling for better coverage:
        - 30% temporal stratification (timeline coverage)
        - 40% activity hotspots (interesting conversations)
        - 30% quality-based (best individual messages)
        """
        if n >= len(self.processed_messages):
            return self.processed_messages
        
        # Allocate samples across strategies
        temporal_count = int(n * 0.30)
        hotspot_count = int(n * 0.40)
        quality_count = int(n * 0.30)
        
        print(f"  📍 Sampling strategy: {temporal_count} temporal + {hotspot_count} hotspots + {quality_count} quality")
        
        # 1. Temporal Stratification
        temporal_samples = self._temporal_stratified_sample(temporal_count)
        
        # 2. Activity Hotspots
        hotspot_samples = self._activity_hotspot_sample(hotspot_count)
        
        # 3. Quality-based
        quality_samples = self._quality_based_sample(quality_count)
        
        # Combine and deduplicate
        all_samples = temporal_samples + hotspot_samples + quality_samples
        seen = set()
        unique = []
        
        for msg in all_samples:
            key = f"{msg['sender']}:{msg['message'][:30]}"
            if key not in seen:
                seen.add(key)
                unique.append(msg)
        
        # Shuffle to mix strategies
        random.shuffle(unique)
        final = unique[:n]
        
        print(f"  ✓ Sampled {len(final)} diverse messages")
        return final
    
    def format_for_llm(self, messages: List[Dict], max_chars: int = 100) -> str:
        """
        Ultra-compact formatting to fit more messages:
        - First name only (max 8 chars)
        - Aggressive truncation
        - Remove extra whitespace
        """
        lines = []
        for msg in messages:
            # Get first name only, limit to 8 chars
            sender = msg['sender'].split()[0][:8]
            
            # Truncate and clean message
            text = msg['message'][:max_chars].strip()
            text = ' '.join(text.split())  # Remove extra whitespace
            
            lines.append(f"{sender}: {text}")
        
        return '\n'.join(lines)
    
    def extract_quick_stats(self) -> Dict:
        """Fast statistical analysis"""
        msg_counts = Counter(msg['sender'] for msg in self.processed_messages)
        msg_lengths = defaultdict(list)
        
        for msg in self.processed_messages:
            msg_lengths[msg['sender']].append(len(msg['message']))
        
        return {
            'message_counts': dict(msg_counts),
            'avg_length': {
                p: round(statistics.mean(msg_lengths[p]), 1) 
                for p in self.participants if msg_lengths[p]
            },
            'total_analyzed': len(self.processed_messages)
        }
    
    def ai_core_analysis(self, sample_size: int = 400) -> Dict:
        """COMBINED: Personality + Roles + Alignments + Relationships"""
        sampled = self.smart_sample(sample_size)
        context = self.format_for_llm(sampled, max_chars=90)
        
        prompt = f"""Analyze WhatsApp chat (English/Hinglish). {len(self.participants)} people: {', '.join(self.participants)}

MESSAGES:
{context}

Provide CONCISE analysis:

1. PERSONALITY (for each person):
   - Style: formal/casual/energetic/calm
   - Tone: supportive/sarcastic/humorous/dramatic
   - 3 key traits

2. TOP ROLES (top scorer for each, 0-100):
   - Therapist: Most supportive
   - Hype Man: Most encouraging
   - Comedian: Funniest
   - Drama Queen: Most dramatic
   - Meme Lord: Best memes
   - Ghost: Lurks then appears
   - Voice of Reason: Most logical
   - Chaos Agent: Creates fun chaos

3. D&D ALIGNMENT (one per person):
   Lawful/Neutral/Chaotic + Good/Neutral/Evil
   Based on: structure vs spontaneity + supportive vs teasing

4. RELATIONSHIPS:
   - Top 2-3 closest pairs
   - Bond type + why

Return ONLY valid JSON:
{{
  "personalities": [{{"name": "X", "style": "...", "tone": "...", "traits": ["a","b","c"]}}],
  "roles": {{"therapist": {{"name": "X", "score": 90, "reason": "brief"}}, "hype_man": {{"name": "X", "score": 85, "reason": "..."}}, ...}},
  "alignments": [{{"name": "X", "alignment": "Chaotic Good", "reason": "brief"}}],
  "pairs": [{{"pair": ["X","Y"], "bond": "type", "reason": "brief"}}]
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a chat analyst. Return ONLY valid JSON, no markdown."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=2500
            )
            
            result = self._clean_json(response.choices[0].message.content)
            parsed = json.loads(result)
            print(f"✓ Core analysis complete: {len(parsed.get('personalities', []))} personalities")
            return parsed
        except Exception as e:
            print(f"⚠️ Core analysis error: {e}")
            return {
                'personalities': [],
                'roles': {},
                'alignments': [],
                'pairs': []
            }
    
    def ai_content_analysis(self, sample_size: int = 500) -> Dict:
        """COMBINED: Topics + Golden Moments + Memes + Humor + Sentiment"""
        sampled = self.smart_sample(sample_size)
        context = self.format_for_llm(sampled, max_chars=95)
        
        prompt = f"""Analyze chat content. {len(self.participants)} people: {', '.join(self.participants)}

MESSAGES:
{context}

Provide detailed analysis:

1. TOP 5 TOPICS:
   - Topic name, frequency (high/medium/low), keywords

2. GOLDEN MOMENTS (8-12 categories, BEST examples only):
   Categories: Most Wholesome, Most Savage, Most Random, Most Inspirational, 
   Most Cringe, Most Relatable, Most Unhinged, Plot Twist, Mic Drop, Big Brain,
   Main Character Energy, Villain Arc
   
   For each:
   - Category name
   - Sender name
   - Actual message text (40-120 chars) - MUST be exact quote
   - Why it's golden (brief)
   - Impact score 75-100

3. MEME ANALYSIS (Find REAL examples):
   a) Dank Moments (3-4): Peak internet humor
   b) Cursed Content (2-3): Unhinged messages  
   c) Copypasta Potential (2-3): Could become memes
   d) Reaction Worthy (2-3): Got big reactions
   e) Inside Jokes (2-3): Recurring references
   f) Ratio Moments (2): Got roasted hard
   g) NPC Energy (2): Predictable responses

4. HUMOR (actual examples):
   - Top 3 funniest moments
   - Top 2 sarcastic burns
   - 2 unintentionally funny

5. SENTIMENT:
   - Mood, energy, vibe

Return ONLY valid JSON:
{{
  "topics": [{{"topic": "Dating Drama", "frequency": "high", "keywords": ["crush","date","relationship"]}}],
  "golden_moments": [{{"category": "Most Savage", "sender": "Alice", "message": "actual quote here", "why": "destroyed Bob's ego", "impact_score": 92}}],
  "memes": {{
    "dank_moments": [{{"sender": "X", "message": "quote", "why_dank": "reason"}}],
    "cursed": [{{"sender": "X", "message": "quote", "cursed_level": "high"}}],
    "copypasta_potential": [{{"sender": "X", "message": "quote", "meme_format": "format"}}],
    "reaction_worthy": [{{"sender": "X", "message": "quote", "reaction": "how others reacted"}}],
    "inside_jokes": [{{"phrase": "catchphrase", "context": "when used", "frequency": "often"}}],
    "ratio_moments": [{{"victim": "X", "roaster": "Y", "message": "the roast"}}],
    "npc_energy": [{{"sender": "X", "pattern": "repeated phrase/behavior"}}]
  }},
  "humor": {{
    "funniest": [{{"sender": "X", "message": "quote", "why": "reason"}}],
    "sarcasm": [{{"sender": "X", "message": "quote", "target": "who"}}],
    "unintentional": [{{"sender": "X", "message": "quote", "why_funny": "reason"}}]
  }},
  "sentiment": {{"mood": "positive", "energy": "high", "vibe": "chaotic friendly"}}
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a meme analyst and content curator. Find the BEST moments. Return ONLY valid JSON, no markdown."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=3500
            )
            
            result = self._clean_json(response.choices[0].message.content)
            parsed = json.loads(result)
            
            # Validate and log
            print(f"✓ Content analysis complete:")
            print(f"  - Topics: {len(parsed.get('topics', []))}")
            print(f"  - Golden moments: {len(parsed.get('golden_moments', []))}")
            print(f"  - Meme categories: {len(parsed.get('memes', {}).keys())}")
            print(f"  - Humor items: {len(parsed.get('humor', {}).get('funniest', []))}")
            
            return parsed
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing error: {e}")
            print(f"Raw response: {result[:200]}...")
            return self._get_empty_content_structure()
        except Exception as e:
            print(f"⚠️ Content analysis error: {e}")
            return self._get_empty_content_structure()
    
    def _get_empty_content_structure(self) -> Dict:
        """Return empty but valid structure"""
        return {
            'topics': [],
            'golden_moments': [],
            'memes': {
                'dank_moments': [],
                'cursed': [],
                'copypasta_potential': [],
                'reaction_worthy': [],
                'inside_jokes': [],
                'ratio_moments': [],
                'npc_energy': []
            },
            'humor': {
                'funniest': [],
                'sarcasm': [],
                'unintentional': []
            },
            'sentiment': {
                'mood': 'unknown',
                'energy': 'unknown',
                'vibe': 'analysis failed'
            }
        }
    
    def enhance_moments_with_context(self, moments: List[Dict]) -> List[Dict]:
        """Add context chat to golden moments"""
        enhanced = []
        for moment in moments:
            try:
                sender = moment.get('sender', '')
                message_text = moment.get('message', '')
                
                if sender and message_text:
                    context = self.get_context_messages(message_text, sender, before=2, after=1)
                    
                    if context:
                        moment['context_chat'] = [
                            {
                                'sender': msg['sender'].split()[0],
                                'message': msg['message'][:150]
                            }
                            for msg in context
                        ]
                    else:
                        moment['context_chat'] = []
                
                enhanced.append(moment)
            except Exception as e:
                print(f"⚠️ Context enhancement error: {e}")
                moment['context_chat'] = []
                enhanced.append(moment)
        
        return enhanced
    
    def _clean_json(self, text: str) -> str:
        """Extract JSON from markdown and clean it"""
        # Remove markdown code blocks
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]
        
        # Find JSON object
        text = text.strip()
        
        # Try to find JSON boundaries
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1:
            text = text[start:end+1]
        
        return text
    
    def generate_report(self) -> Dict:
        """Generate optimized analysis report"""
        print("\n" + "="*60)
        print("ENHANCED CHAT ANALYSIS v3.1")
        print("="*60 + "\n")
        start = time.time()
        
        report = {
            'metadata': {
                'total_messages': len(self.processed_messages),
                'participants': self.participants,
                'analysis_timestamp': datetime.now().isoformat(),
                'optimization': 'v3.1 - Multi-strategy sampling with hotspots'
            }
        }
        
        # Quick stats (no API call)
        print("📊 Quick Stats...")
        report['stats'] = self.extract_quick_stats()
        
        # API Call 1: Core Analysis (400 messages now)
        print("🤖 API Call [1/2]: Core Analysis (400 samples)...")
        core = self.ai_core_analysis(400)
        report['personalities'] = core.get('personalities', [])
        report['roles'] = core.get('roles', {})
        report['alignments'] = core.get('alignments', [])
        report['closest_pairs'] = core.get('pairs', [])
        time.sleep(0.5)
        
        # API Call 2: Content Analysis with Memes (500 messages now)
        print("🤖 API Call [2/2]: Content & Meme Analysis (500 samples)...")
        content = self.ai_content_analysis(500)
        report['topics'] = content.get('topics', [])
        
        # Enhance golden moments with context
        golden_moments = content.get('golden_moments', [])
        if golden_moments:
            print(f"📝 Adding context to {len(golden_moments)} golden moments...")
            report['golden_moments'] = self.enhance_moments_with_context(golden_moments)
        else:
            report['golden_moments'] = []
        
        report['memes'] = content.get('memes', {})
        report['humor'] = content.get('humor', {})
        report['sentiment'] = content.get('sentiment', {})
        
        elapsed = time.time() - start
        report['metadata']['analysis_time'] = round(elapsed, 2)
        
        print(f"\n✅ Complete in {elapsed:.1f}s")
        print(f"   Topics: {len(report.get('topics', []))}")
        print(f"   Golden Moments: {len(report.get('golden_moments', []))}")
        print(f"   Meme Categories: {len(report.get('memes', {}).keys())}\n")
        
        return report


# ===== FLASK API =====

@app.route('/health', methods=['GET', 'OPTIONS'])
def health_check():
    if request.method == 'OPTIONS':
        return '', 204
    return jsonify({
        'status': 'healthy',
        'service': 'Enhanced WhatsApp Analyzer',
        'version': '3.1-improved-sampling',
        'features': ['multi_strategy_sampling', 'hotspot_detection', 'context_chat', 'extended_memes']
    })


@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_chat():
    """Main analysis endpoint (enhanced)"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        
        if 'chat_data' not in data:
            return jsonify({'error': 'Missing chat_data'}), 400
        
        chat_data = data['chat_data']
        
        required = ['messages', 'participants']
        for field in required:
            if field not in chat_data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        api_key = data.get('api_key', DEFAULT_GROQ_API_KEY)
        if not api_key:
            return jsonify({'error': 'No API key provided'}), 400
        
        print(f"\n{'='*60}")
        print(f"ANALYSIS REQUEST")
        print(f"Messages: {len(chat_data['messages'])}")
        print(f"Participants: {len(chat_data['participants'])}")
        print(f"Mode: Enhanced v3.1 (multi-strategy sampling)")
        print(f"{'='*60}")
        
        analyzer = OptimizedWhatsAppAnalyzer(chat_data, api_key)
        report = analyzer.generate_report()
        
        return jsonify({'success': True, 'report': report})
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Analysis failed', 'details': str(e)}), 500


@app.route('/analyze/ultra-quick', methods=['POST', 'OPTIONS'])
def ultra_quick():
    """Ultra-fast analysis (1 API call, 250 messages with improved sampling)"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        if 'chat_data' not in data:
            return jsonify({'error': 'Missing chat_data'}), 400
        
        chat_data = data['chat_data']
        api_key = data.get('api_key', DEFAULT_GROQ_API_KEY)
        
        if not api_key:
            return jsonify({'error': 'No API key'}), 400
        
        analyzer = OptimizedWhatsAppAnalyzer(chat_data, api_key)
        
        report = {
            'metadata': {
                'total_messages': len(analyzer.processed_messages),
                'participants': analyzer.participants,
                'analysis_type': 'ultra-quick-v3.1'
            }
        }
        
        report['stats'] = analyzer.extract_quick_stats()
        core = analyzer.ai_core_analysis(250)
        report['personalities'] = core.get('personalities', [])
        report['roles'] = core.get('roles', {})
        report['alignments'] = core.get('alignments', [])
        
        return jsonify({'success': True, 'report': report})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Ultra-quick failed', 'details': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("WhatsApp Analyzer v3.1 - Improved Sampling")
    print("Features:")
    print("  • Multi-strategy sampling (temporal + hotspots + quality)")
    print("  • Activity hotspot detection")
    print("  • Better conversation thread coverage")
    print("  • Increased samples: 400 core + 500 content")
    print("  • Ultra-compact formatting (90-95 chars/msg)")
    print("="*60 + "\n")
    
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=False)