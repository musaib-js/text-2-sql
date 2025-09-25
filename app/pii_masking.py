import re
from cryptography.fernet import Fernet

class PIIMasker:
    def __init__(self, nlp, secret_key: str = None):
        self.key = secret_key.encode() if secret_key else Fernet.generate_key()
        self.cipher = Fernet(self.key)
        self.pii_map = {} 
        self.nlp = nlp
    
    def _encrypt(self, text: str) -> str:
        return self.cipher.encrypt(text.encode()).decode()
    
    def _decrypt(self, token: str) -> str:
        return self.cipher.decrypt(token.encode()).decode()
    
    def mask(self, query: str) -> str:
        """Detect & mask PII using NER + regex"""
        self.pii_map.clear()
        placeholder_count = 1
        masked_query = query
        
        patterns = {
            "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "PHONE": r"\+?\d[\d\-\s]{7,}\d",
            "DOB ONE": r"\b\d{4}-\d{2}-\d{2}\b",
            "DOB TWO": r"\b\d{2}/\d{2}/\d{4}\b",
        }
        
        for label, pattern in patterns.items():
            matches = re.findall(pattern, masked_query)
            for match in matches:
                placeholder = f"<PII_{placeholder_count}>"
                encrypted = self._encrypt(match)
                self.pii_map[placeholder] = encrypted
                masked_query = masked_query.replace(match, placeholder)
                placeholder_count += 1
        
        # --- Step 2: NER for names ---
        doc = self.nlp(masked_query)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                match = ent.text
                placeholder = f"<PII_{placeholder_count}>"
                encrypted = self._encrypt(match)
                self.pii_map[placeholder] = encrypted
                masked_query = masked_query.replace(match, placeholder)
                placeholder_count += 1
        
        return masked_query
    
    def unmask(self, text: str) -> str:
        """Replace placeholders with decrypted PII"""
        unmasked_text = text
        for placeholder, encrypted in self.pii_map.items():
            original = self._decrypt(encrypted)
            unmasked_text = unmasked_text.replace(placeholder, original)
        return unmasked_text
