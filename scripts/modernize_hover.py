import re

def update_css():
    with open('static/css/style.css', 'r', encoding='utf-8') as f:
        css = f.read()
        
    # 1. Add variables to :root
    root_vars = """
    /* Modern Hover System */
    --hover-translate-y: -3px;
    --hover-shadow-modern: 0 10px 20px -5px rgba(0, 0, 0, 0.08);
    --hover-transition: transform 0.25s cubic-bezier(0.25, 0.46, 0.45, 0.94), box-shadow 0.25s cubic-bezier(0.25, 0.46, 0.45, 0.94), border-color 0.25s ease;
"""
    if '--hover-translate-y' not in css:
        css = re.sub(r'(:root\s*\{[^}]*)(\})', lambda m: m.group(1) + root_vars + m.group(2), css, count=1, flags=re.DOTALL)
        
    # 2. Add dark mode variables
    dark_vars = """
    --hover-shadow-modern: 0 10px 20px -5px rgba(0, 0, 0, 0.35);
"""
    if '10px 20px -5px rgba(0, 0, 0, 0.35)' not in css:
        css = re.sub(r'(\[data-theme="dark"\]\s*\{[^}]*)(\})', lambda m: m.group(1) + dark_vars + m.group(2), css, count=1, flags=re.DOTALL)

    # 3. Replace .feature-card:hover and related overly dramatic animations
    # We will just replace the entire block for .feature-card:hover and its children
    
    # Feature card hover replacement
    css = re.sub(r'\.feature-card:hover\s*\{[^}]*\}',
                 '.feature-card:hover {\n    transform: translateY(var(--hover-translate-y));\n    box-shadow: var(--hover-shadow-modern);\n    border-color: var(--success-border);\n    cursor: pointer;\n}', css)
                 
    css = re.sub(r'\.feature-card:hover::before\s*\{[^}]*\}',
                 '.feature-card:hover::before {\n    left: 100%;\n    opacity: 0.5;\n}', css)
                 
    css = re.sub(r'\.feature-card:hover\s+\.feature-icon-wrapper\s*\{[^}]*\}',
                 '.feature-card:hover .feature-icon-wrapper {\n    transform: translateY(-2px);\n    background: var(--emerald-green);\n    color: white;\n}', css)
                 
    css = re.sub(r'\.feature-card:hover\s+i,\s*\.feature-card:hover\s+svg\.lucide\s*\{[^}]*\}',
                 '.feature-card:hover i, .feature-card:hover svg.lucide {\n    transform: scale(1.05);\n    color: white;\n}', css)

    # Update .feature-card base transition
    css = re.sub(r'transition:\s*transform\s*0\.4s\s*ease,\s*box-shadow\s*0\.4s\s*ease,\s*border-color\s*0\.4s\s*ease,\s*background\s*0\.4s\s*ease;', 
                 'transition: var(--hover-transition), background 0.4s ease;', css)
                 
    # 4. Create universal hover card class
    hover_card_class = """
/* Modern Card Hover System */
.modern-hover {
    transition: var(--hover-transition) !important;
    will-change: transform, box-shadow;
}

.modern-hover:hover {
    transform: translateY(var(--hover-translate-y)) !important;
    box-shadow: var(--hover-shadow-modern) !important;
}

/* Reduced Motion Support */
@media (prefers-reduced-motion: reduce) {
    .modern-hover,
    .feature-card,
    .dashboard-card,
    .glass-card,
    .comparison-result-card,
    .upload-card {
        transition: none !important;
    }
    
    .modern-hover:hover,
    .feature-card:hover,
    .dashboard-card:hover,
    .glass-card:hover,
    .comparison-result-card:hover,
    .upload-card:hover {
        transform: none !important;
        box-shadow: none !important;
        /* Fallback: just adjust border or background lightly */
        filter: brightness(0.95);
    }
    
    [data-theme="dark"] .modern-hover:hover,
    [data-theme="dark"] .feature-card:hover,
    [data-theme="dark"] .dashboard-card:hover,
    [data-theme="dark"] .glass-card:hover,
    [data-theme="dark"] .comparison-result-card:hover,
    [data-theme="dark"] .upload-card:hover {
        filter: brightness(1.1);
    }
}
"""
    if '/* Modern Card Hover System */' not in css:
        css += hover_card_class
        
    # We also need to search for any other hardcoded box-shadows or transforms on hover and soften them
    # For instance: transform: translateY(-10px) 
    css = re.sub(r'transform:\s*translateY\(-10px\)\s*;', 'transform: translateY(var(--hover-translate-y));', css)
    css = re.sub(r'transform:\s*translateY\(-8px\)\s*;', 'transform: translateY(var(--hover-translate-y));', css)
    css = re.sub(r'transform:\s*translateY\(-5px\)\s*;', 'transform: translateY(var(--hover-translate-y));', css)
    
    # We should add `.modern-hover` to all generic card definitions
    # Actually, applying `.modern-hover` automatically via CSS is hard, we can just map the specific cards
    card_types = [
        '.dashboard-card',
        '.upload-card',
        '.tip-card',
        '.help-card',
        '.result-card',
        '.analysis-card',
        '.demo-card',
        '.ai-summary-card',
        '.comparison-card',
        '.story-card',
        '.insight-card',
        '.recommendation-card'
    ]
    
    # Instead of modifying HTML, we can just append a rule that groups these:
    grouped_hover = ",\n".join(card_types) + " {\n    transition: var(--hover-transition);\n    will-change: transform, box-shadow;\n}\n"
    grouped_hover += ",\n".join([c + ":hover" for c in card_types]) + " {\n    transform: translateY(var(--hover-translate-y));\n    box-shadow: var(--hover-shadow-modern);\n}\n"
    
    if "/* Apply to specific cards */" not in css:
        css += "\n/* Apply to specific cards */\n" + grouped_hover

    with open('static/css/style.css', 'w', encoding='utf-8') as f:
        f.write(css)
    print("style.css modernized successfully")

if __name__ == '__main__':
    update_css()
