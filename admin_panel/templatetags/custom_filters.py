from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """
    Custom template filter to access dictionary items using key
    Usage: {{ dictionary|get_item:key }}
    """
    return dictionary.get(key) 