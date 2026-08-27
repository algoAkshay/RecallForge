"""Streamlit-native theme refinements for the RecallForge workspace."""


def workspace_css() -> str:
    """Refine Streamlit's resolved theme without maintaining a second theme state."""
    return '''<style>
:root{--rf-bg:var(--background-color);--rf-surface:var(--background-color);--rf-surface-secondary:var(--secondary-background-color);--rf-sidebar:color-mix(in srgb,var(--secondary-background-color) 88%,var(--background-color));--rf-text:var(--text-color);--rf-text-secondary:color-mix(in srgb,var(--text-color) 68%,var(--background-color));--rf-border:var(--border-color);--rf-hover:color-mix(in srgb,var(--secondary-background-color) 82%,var(--text-color));--rf-input-bg:var(--secondary-background-color);--rf-code-bg:var(--code-background-color,var(--secondary-background-color));--rf-shadow:0 1px 2px color-mix(in srgb,var(--text-color) 10%,transparent);--rf-link:var(--primary-color);--rf-web:var(--primary-color);--rf-memory:var(--green-color,var(--primary-color));--rf-focus:var(--primary-color)}
.stApp{font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
[data-testid="stAppViewContainer"],.stApp p,.stApp li,.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5,.stApp h6,.stApp strong,.stApp em,.stApp blockquote{color:var(--rf-text)}
[data-testid="stSidebar"]{background-color:var(--rf-sidebar);border-right:1px solid var(--rf-border)}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label{color:var(--rf-text)}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],.stCaption{color:var(--rf-text-secondary)}
.block-container{max-width:1050px;padding-top:2rem;padding-bottom:6rem}
.rf-kicker,.rf-reason{color:var(--rf-text-secondary)!important;font-size:.86rem}.rf-status{font-size:.78rem;font-weight:700;letter-spacing:.05em}.rf-memory{color:var(--rf-memory)!important}.rf-web{color:var(--rf-web)!important}
a,.stApp a{color:var(--rf-link)!important}[data-testid="stChatMessage"]{background:transparent;border:0;padding:.25rem 0;color:var(--rf-text)}
[data-testid="stChatInput"],[data-testid="stChatInput"]>div,textarea,input,[data-baseweb="select"]>div,[data-baseweb="input"]>div{background-color:var(--rf-input-bg)!important;color:var(--rf-text)!important;border-color:var(--rf-border)!important;box-shadow:var(--rf-shadow)!important}
textarea::placeholder,input::placeholder{color:var(--rf-text-secondary)!important;opacity:1}
textarea:focus,input:focus,[data-baseweb="select"]>div:focus-within,[data-baseweb="input"]>div:focus-within{border-color:var(--rf-focus)!important;box-shadow:0 0 0 2px color-mix(in srgb,var(--rf-focus) 18%,transparent)!important}
[data-baseweb="checkbox"]>div{border-color:var(--rf-border)!important;background-color:var(--rf-input-bg)!important}
.stButton>button,[data-testid="stDownloadButton"]>button{border-radius:7px;border:1px solid var(--rf-border);background-color:var(--rf-surface);color:var(--rf-text)}
.stButton>button:hover,[data-testid="stDownloadButton"]>button:hover{background-color:var(--rf-hover);border-color:var(--rf-focus);color:var(--rf-text)}
.stButton>button:disabled{color:var(--rf-text-secondary)!important;opacity:.65}
[data-testid="stSidebar"] .stButton>button{background-color:transparent}[data-testid="stSidebar"] .stButton>button:hover{background-color:var(--rf-hover)}
[data-testid="stExpander"],[data-testid="stPopoverBody"],[data-testid="stDialog"],div[data-baseweb="popover"],ul[role="listbox"],div[role="dialog"]{background-color:var(--rf-surface)!important;color:var(--rf-text)!important;border-color:var(--rf-border)!important;box-shadow:var(--rf-shadow)!important}
[data-testid="stDialog"]{opacity:1!important;background-color:var(--secondary-background-color)!important;color:var(--text-color)!important;border:1px solid var(--border-color)!important;border-radius:14px;padding:.5rem;box-shadow:var(--rf-shadow)!important}
[class*="st-key-recent_thread_row_"]{margin-bottom:.3rem;border-radius:10px;padding:.1rem .2rem;transition:background-color 140ms ease,color 140ms ease}
[class*="st-key-recent_thread_row_"]:hover,[class*="st-key-recent_thread_row_active_"]{background-color:var(--rf-hover)}
[class*="st-key-thread_title_"] .stButton>button,[class*="st-key-thread_menu_"] .stButton>button{background-color:transparent!important;border-color:transparent!important;box-shadow:none!important;color:var(--rf-text)!important}
[class*="st-key-thread_title_"] .stButton>button{justify-content:flex-start;min-width:0;padding:.35rem .4rem;text-align:left}
[class*="st-key-thread_title_"] .stButton>button p{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
[class*="st-key-thread_title_"] .stButton>button:disabled{color:var(--rf-text)!important;opacity:1}
[class*="st-key-thread_menu_"] .stButton>button{border-radius:8px;min-width:2rem;padding:.3rem .4rem}
[class*="st-key-thread_menu_"] .stButton>button:hover{background-color:var(--rf-hover)!important;border-color:transparent!important}
[data-baseweb="menu"] [role="menuitem"],ul[role="listbox"] li{color:var(--rf-text)!important}[data-baseweb="menu"] [role="menuitem"]:hover,ul[role="listbox"] li:hover{background-color:var(--rf-hover)!important}
[data-testid="stVerticalBlockBorderWrapper"],div[data-testid="stToast"]{background-color:var(--rf-surface);border-color:var(--rf-border);box-shadow:var(--rf-shadow)}
table{display:block;overflow-x:auto;border-collapse:collapse;color:var(--rf-text)}th,td{color:var(--rf-text)!important;border:1px solid var(--rf-border)!important;padding:.45rem .65rem!important}th{background-color:var(--rf-surface-secondary)!important}
code{color:var(--rf-text)!important;background-color:var(--rf-code-bg)!important}pre{color:var(--rf-text)!important;background-color:var(--rf-code-bg)!important;border:1px solid var(--rf-border);border-radius:8px}
[data-testid="stAppViewContainer"],[data-testid="stSidebar"],[data-testid="stChatInput"],[data-baseweb="select"]>div,[data-baseweb="input"]>div,.stButton>button,[data-testid="stDownloadButton"]>button,[data-testid="stExpander"],[data-testid="stPopoverBody"],[data-testid="stVerticalBlockBorderWrapper"]{transition:background-color 160ms ease,color 160ms ease,border-color 160ms ease,box-shadow 160ms ease}
</style>'''
