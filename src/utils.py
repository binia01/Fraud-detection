import numpy as np

def ip_to_int(ip_series):
    """Converts a pandas series of string IP addresses to integers"""
    def _convert(ip):
        try:
            # Case 1: If it's already a number (float or int), just cast to int
            if isinstance(ip, (int, float, np.number)):
                return int(ip)
            
            # Case 2: If it's a string representation of a number ("732758368")
            if isinstance(ip, str) and ip.replace('.', '').isdigit() and '.' not in ip:
                 return int(ip)

            # Case 3: If it's a standard IPv4 string ("192.168.0.1")
            if isinstance(ip, str) and '.' in ip:
                parts = [int(p) for p in ip.split('.')]
                if len(parts) == 4:
                    return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
            
            # Fallback
            return 0
        except:
            return 0
        
    return ip_series.apply(_convert)

