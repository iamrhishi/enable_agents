from flask import Blueprint, request, jsonify
from nlp_api_caller import NLPProcessor

nlp_api = Blueprint('nlp_api', __name__)

# Initialize processor (auto-detects config.json and .env)
processor = NLPProcessor()

@nlp_api.route('/api/nlp-chat', methods=['POST'])
def nlp_chat():
    try:
        data = request.json
        user_query = data.get('query')
        if not user_query:
            return jsonify({'success': False, 'error': 'No query provided'}), 400
        result = processor.process(user_query)
        return jsonify({
            'success': True,
            'summary': result.get('summary'),
            'data': result.get('data')
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
