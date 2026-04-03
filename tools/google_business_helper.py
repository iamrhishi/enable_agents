"""
Google My Business Integration Helper Module
Handles credential management and API interactions with Google Business Profile API
"""

import os
import json
import requests
from dotenv import load_dotenv
from datetime import datetime

class GoogleBusinessHelper:
    """Helper class for Google My Business API interactions"""
    
    BASE_URL = "https://mybusiness.googleapis.com/v4"
    
    def __init__(self):
        """Initialize with credentials from environment"""
        load_dotenv()
        self.client_id = os.getenv('GOOGLE_CLIENT_ID')
        self.client_secret = os.getenv('GOOGLE_CLIENT_SECRET')
        self.access_token = None
    
    def is_connected(self) -> bool:
        """Check if Google Business credentials are configured"""
        return bool(self.client_id and self.client_secret)
    
    def get_business_profile(self) -> dict:
        """Fetch complete business profile information"""
        if not self.is_connected():
            return {'error': 'Not connected', 'code': 'NOT_CONNECTED'}
        
        try:
            # This would use real API in production
            # For now, returns structure for future implementation
            profile_url = f"{self.BASE_URL}/accounts/{self.account_id}/locations/{self.location_id}"
            
            # TODO: Implement actual API call with OAuth token
            # response = requests.get(
            #     profile_url,
            #     headers={'Authorization': f'Bearer {self.access_token}'}
            # )
            
            return {
                'name': 'Your Business Name',
                'address': '123 Main St, City, ST 12345',
                'phone': '+1-555-123-4567',
                'website': 'https://yourbusiness.com',
                'email': 'contact@yourbusiness.com'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_business_insights(self) -> dict:
        """Fetch business performance insights and metrics"""
        if not self.is_connected():
            return {'error': 'Not connected', 'code': 'NOT_CONNECTED'}
        
        try:
            insights_url = f"{self.BASE_URL}/accounts/{self.account_id}/locations/{self.location_id}/insights"
            
            # TODO: Implement actual API call with OAuth token
            # response = requests.get(
            #     insights_url,
            #     headers={'Authorization': f'Bearer {self.access_token}'}
            # )
            
            return {
                'views': {
                    'total': 2450,
                    'monthlyChange': '+12%'
                },
                'actions': {
                    'phoneClicks': 342,
                    'directionRequests': 567,
                    'websiteClicks': 234
                },
                'photos': {
                    'totalPhotos': 24,
                    'monthlyViews': 1205
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_reviews(self, limit: int = 10) -> dict:
        """Fetch customer reviews and ratings"""
        if not self.is_connected():
            return {'error': 'Not connected', 'code': 'NOT_CONNECTED'}
        
        try:
            reviews_url = f"{self.BASE_URL}/accounts/{self.account_id}/locations/{self.location_id}/reviews"
            
            # TODO: Implement actual API call
            # response = requests.get(
            #     reviews_url + f'?maxResults={limit}',
            #     headers={'Authorization': f'Bearer {self.access_token}'}
            # )
            
            return {
                'totalReviews': 156,
                'averageRating': 4.5,
                'recentReviews': [
                    {
                        'author': 'John Doe',
                        'rating': 5,
                        'text': 'Excellent service and great experience!',
                        'date': '2025-03-08'
                    },
                    {
                        'author': 'Jane Smith',
                        'rating': 4,
                        'text': 'Good quality, could improve wait times',
                        'date': '2025-03-07'
                    }
                ]
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_posts(self, limit: int = 5) -> dict:
        """Fetch recent business posts"""
        if not self.is_connected():
            return {'error': 'Not connected', 'code': 'NOT_CONNECTED'}
        
        try:
            posts_url = f"{self.BASE_URL}/accounts/{self.account_id}/locations/{self.location_id}/posts"
            
            # TODO: Implement actual API call
            # response = requests.get(
            #     posts_url + f'?maxResults={limit}',
            #     headers={'Authorization': f'Bearer {self.access_token}'}
            # )
            
            return {
                'totalPosts': 28,
                'recentPosts': [
                    {
                        'title': 'Spring Menu Launch',
                        'description': 'Check out our new spring menu items',
                        'date': '2025-03-01'
                    }
                ]
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_messaging_status(self) -> dict:
        """Check messaging/Q&A status"""
        if not self.is_connected():
            return {'error': 'Not connected', 'code': 'NOT_CONNECTED'}
        
        return {
            'messagingEnabled': True,
            'respondToMessagesEnabled': True,
            'averageResponseTime': '2 hours',
            'unreadCount': 3
        }
    
    def get_complete_business_data(self) -> dict:
        """Fetch all business data in one call"""
        if not self.is_connected():
            return {
                'connected': False,
                'error': 'Google Business not connected'
            }
        
        return {
            'connected': True,
            'businessProfile': self.get_business_profile(),
            'insights': self.get_business_insights(),
            'reviews': self.get_reviews(),
            'posts': self.get_posts(),
            'messaging': self.get_messaging_status(),
            'fetchedAt': datetime.now().isoformat()
        }
    
    def save_credentials(self, credentials: dict) -> bool:
        """Save credentials to environment variables"""
        try:
            env_vars = {
                'GOOGLE_CLIENT_ID': credentials.get('clientId'),
                'GOOGLE_CLIENT_SECRET': credentials.get('clientSecret'),
                'GOOGLE_REDIRECT_URI': credentials.get('redirectUri')
            }
            
            # Update environment
            for key, value in env_vars.items():
                os.environ[key] = value
            
            # Reload .env file
            load_dotenv()
            
            return True
        except Exception as e:
            print(f"Error saving credentials: {str(e)}")
            return False
    
    def validate_credentials(self) -> bool:
        """Validate that all required credentials are present"""
        required_fields = [
            'GOOGLE_CLIENT_ID',
            'GOOGLE_CLIENT_SECRET',
            'GOOGLE_REDIRECT_URI'
        ]
        
        return all(os.getenv(field) for field in required_fields)


class GoogleBusinessSearcher:
    """Search class using Google Locations API to find businesses"""
    
    def __init__(self):
        """Initialize searcher with Google credentials"""
        load_dotenv()
        self.client_id = os.getenv('GOOGLE_CLIENT_ID')
        self.client_secret = os.getenv('GOOGLE_CLIENT_SECRET')
        self.access_token = None
        self.connection_established = False
    
    def set_credentials(self, access_token: str):
        """
        Set Google OAuth access token for API calls
        
        Args:
            access_token: Valid Google OAuth access token
        """
        self.access_token = access_token
        self.connection_established = True
    
    def search_businesses(self, query: str, location: str, max_results: int = 200, page: int = 1, page_size: int = 20) -> dict:
        """
        Search for businesses using Google Places API (New) with pagination.
        Args:
            query: Business type/keyword to search for (e.g., "fencing", "restaurants")
            location: City/region to search in (e.g., "Florida", "San Francisco")
            max_results: Maximum number of businesses to return (default 200, capped at 200)
            page: Page number for pagination (1-indexed, default 1)
            page_size: Number of results per page (default 20, max 200)
        Returns:
            Dictionary with paginated search results
        """
        try:
            api_key = os.getenv('GOOGLE_PLACES_API_KEY')
            if not api_key:
                return {
                    'success': False,
                    'error': 'Google Places API key not configured.',
                    'code': 'API_KEY_MISSING'
                }
            
            # Cap max_results at 200
            if max_results > 200:
                max_results = 200
            
            # Cap page_size at 200
            if page_size > 200:
                page_size = 200
            
            api_endpoint = "https://places.googleapis.com/v1/places:searchText"
            headers = {
                'Content-Type': 'application/json',
                'X-Goog-FieldMask': 'places.displayName,places.formattedAddress,places.internationalPhoneNumber,places.websiteUri,places.location,places.rating,places.userRatingCount,places.id,nextPageToken'
            }
            
            # Create variations of the query to bypass the 60-result limit per query
            base_query = f"{query} in {location}" if location else query
            query_variations = [
                base_query,
                f"top {query} in {location}" if location else f"top {query}",
                f"best {query} in {location}" if location else f"best {query}",
                f"{query} in North {location}" if location else f"{query} in North",
                f"{query} in South {location}" if location else f"{query} in South"
            ]
            
            print(f"[PLACES_API] Searching with variations to target Max Results: {max_results}")
            
            all_businesses = []
            seen_place_ids = set()
            api_page_size = min(60, page_size)  # Google API API page limit
            
            # Fetch across all variations until max_results is reached
            for current_query in query_variations:
                if len(all_businesses) >= max_results:
                    break
                    
                print(f"[PLACES_API] Running query: '{current_query}'")
                next_page_token = None
                
                while len(all_businesses) < max_results:
                    fetch_size = min(api_page_size, max_results - len(all_businesses))
                    payload = {
                        'textQuery': current_query,
                        'pageSize': fetch_size,
                        'languageCode': 'en'
                    }
                    if next_page_token:
                        payload['pageToken'] = next_page_token
                    
                    response = requests.post(
                        f"{api_endpoint}?key={api_key}",
                        headers=headers,
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code != 200:
                        print(f"[PLACES_API] Error {response.status_code}: {response.text}")
                        # If a variation fails, break and try the next variation instead of failing the whole request
                        break
                    
                    api_results = response.json()
                    places = api_results.get('places', [])
                    
                    for place in places:
                        place_id = place.get('id', '')
                        if not place_id or place_id in seen_place_ids:
                            continue
                            
                        seen_place_ids.add(place_id)
                        all_businesses.append({
                            'id': place_id,
                            'name': place.get('displayName', {}).get('text', ''),
                            'address': place.get('formattedAddress', ''),
                            'phone': place.get('internationalPhoneNumber', ''),
                            'website': place.get('websiteUri', ''),
                            'description': '',
                            'latitude': place.get('location', {}).get('latitude'),
                            'longitude': place.get('location', {}).get('longitude'),
                            'rating': place.get('rating'),
                            'userRatingCount': place.get('userRatingCount'),
                            'matchAccuracy': 'high' if place.get('rating') else 'medium',
                            'isPrimary': False,
                            'placeId': place_id
                        })
                        
                        if len(all_businesses) >= max_results:
                            break
                    
                    next_page_token = api_results.get('nextPageToken')
                    
                    # If this specific query variation ran out of pages/results, move to the next variation
                    if not next_page_token or len(places) == 0:
                        break
            
            print(f"[PLACES_API] Found {len(all_businesses)} total results")
            
            # Apply pagination to results
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated_businesses = all_businesses[start_idx:end_idx]
            
            # Calculate pagination metadata
            total_pages = (len(all_businesses) + page_size - 1) // page_size
            has_next = page < total_pages
            has_prev = page > 1
            
            return {
                'success': True,
                'searchQuery': query,
                'location': location,
                'totalResults': len(all_businesses),
                'resultsThisPage': len(paginated_businesses),
                'businesses': paginated_businesses,
                'pagination': {
                    'page': page,
                    'pageSize': page_size,
                    'totalPages': total_pages,
                    'hasNext': has_next,
                    'hasPrev': has_prev,
                    'startIndex': start_idx + 1,
                    'endIndex': min(end_idx, len(all_businesses))
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'Google API request timed out'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


class GoogleBusinessAnalyzer:
    """Analyzer class for business insights integration with requirements"""
    
    def __init__(self, helper: GoogleBusinessHelper):
        self.helper = helper
    
    def enrich_requirements_context(self, requirements: dict) -> dict:
        """
        Enrich requirement data with business context from Google Business
        
        Args:
            requirements: Dictionary with user requirements
        
        Returns:
            Enriched requirements with business context
        """
        if not self.helper.is_connected():
            return requirements
        
        # Get business data
        business_data = self.helper.get_complete_business_data()
        
        # Add business context to requirements
        enriched = requirements.copy()
        enriched['businessContext'] = {
            'businessName': business_data.get('businessProfile', {}).get('name'),
            'rating': business_data.get('reviews', {}).get('averageRating'),
            'reviewCount': business_data.get('reviews', {}).get('totalReviews'),
            'monthlyViews': business_data.get('insights', {}).get('views', {}).get('total'),
            'customerActions': business_data.get('insights', {}).get('actions', {}),
            'messagingEnabled': business_data.get('messaging', {}).get('messagingEnabled')
        }
        
        return enriched
    
    def generate_business_insights_prompt(self) -> str:
        """
        Generate a prompt segment with business-specific insights
        for AI processing
        """
        if not self.helper.is_connected():
            return ""
        
        data = self.helper.get_complete_business_data()
        insights = data.get('insights', {})
        reviews = data.get('reviews', {})
        
        prompt = f"""
Business Performance Context:
- Profile Views: {insights.get('views', {}).get('total')} monthly views
- Customer Actions: {insights.get('actions', {}).get('phoneClicks')} phone clicks, {insights.get('actions', {}).get('directionRequests')} direction requests
- Customer Rating: {reviews.get('averageRating')}/5 stars ({reviews.get('totalReviews')} reviews)
- Photo Engagement: {insights.get('photos', {}).get('monthlyViews')} monthly photo views

This business data should inform the recommendations provided.
        """
        return prompt.strip()


# Example usage
if __name__ == "__main__":
    helper = GoogleBusinessHelper()
    
    print(f"Connected: {helper.is_connected()}")
    print(f"Valid: {helper.validate_credentials()}")
    
    if helper.is_connected():
        data = helper.get_complete_business_data()
        print(json.dumps(data, indent=2))
