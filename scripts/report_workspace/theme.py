"""A consistent research interface for audio, visual and behavior workflows."""
HERO='''<div class="research-hero"><div><div class="research-eyebrow">EXPLAINABLE MULTIMODAL RESEARCH</div><h1>{title}</h1><p>{subtitle}</p><div class="research-tags"><span>Audio signals</span><span>Visual features</span><span>Behavior measurements</span></div></div><div class="signal-art" aria-hidden="true"><svg viewBox="0 0 300 200"><rect x="155" y="30" width="115" height="85" rx="13" fill="none" stroke="#b9afff" stroke-width="3"/><rect x="168" y="43" width="89" height="59" rx="6" fill="none" stroke="#b9afff" opacity=".4"/><path d="M15 130h15l8-24 12 48 12-75 12 103 12-72 12 38 10-18h35" fill="none" stroke="#72e5d8" stroke-width="4" stroke-linejoin="round"/><circle cx="194" cy="149" r="9" fill="#ffc675"/><circle cx="239" cy="166" r="9" fill="#ffc675"/><circle cx="254" cy="131" r="9" fill="#ffc675"/><path d="M194 149l45 17 15-35" fill="none" stroke="#ffc675" stroke-width="2"/></svg><div>Inspect signals. Explain outputs.</div></div></div>'''
def hero(st,title,subtitle):
    from html import escape
    st.markdown(HERO.format(title=escape(title),subtitle=escape(subtitle)),unsafe_allow_html=True)
def apply_theme(st):
    st.markdown('''<style>
    .stApp{background:#f3f6fa;color:#17334b}
    header[data-testid="stHeader"]{background:#f3f6fa!important}
    .block-container{max-width:1220px;padding-top:1.8rem;padding-bottom:3rem}
    h1,h2,h3{color:#17334b;letter-spacing:-.025em}
    [data-testid="stSidebar"]{background:#142f46!important;border-right:1px solid #23445d}
    [data-testid="stSidebar"] *, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"], [data-testid="stSidebar"] a,[data-testid="stSidebar"] label{color:#edf5fb!important}
    [data-testid="stSidebar"] [aria-current="page"]{background:#294b65;border-radius:9px}
    .block-container [data-testid="stMarkdownContainer"]{color:#17334b}
    .block-container label{color:#375368}
    .research-hero{display:grid;grid-template-columns:1fr 270px;gap:25px;align-items:center;background:#17334b;border-radius:22px;padding:34px 38px;margin-bottom:20px}
    .research-hero h1{color:#fff!important;margin:8px 0 14px;font-size:2.7rem;line-height:1.12}
    .research-hero p{color:#d8e6ef!important;max-width:700px;font-size:1.03rem;line-height:1.65}
    .research-eyebrow{color:#77e1d5;font-size:.72rem;letter-spacing:.16em;font-weight:700}
    .research-tags{display:flex;flex-wrap:wrap;gap:8px;margin-top:20px}
    .research-tags span{border:1px solid #4b6b80;border-radius:24px;padding:6px 11px;color:#edf7fb;font-size:.75rem}
    .research-tags span:nth-child(2){border-color:#8276ac;color:#dcd4ff}
    .research-tags span:nth-child(3){border-color:#aa8656;color:#ffe0af}
    .signal-art svg{width:100%;height:auto}.signal-art div{font-size:.7rem;color:#b8cedd;text-align:center;letter-spacing:.08em}
    .research-card{background:#fff;border:1px solid #d7e2eb;border-radius:16px;padding:24px;min-height:185px;margin-bottom:12px}
    .research-card h3{font-size:1.2rem;margin:10px 0;color:#17334b}.research-card p{color:#536d80;font-size:.9rem;line-height:1.6;margin:0}
    .research-card .tag{font-size:.68rem;letter-spacing:.12em;color:#087f8c;font-weight:700}
    [data-testid="stMetric"]{background:#fff;border:1px solid #d7e2eb;border-radius:14px;padding:16px}
    .block-container button[kind="secondary"],.block-container [data-testid="stBaseButton-secondary"]{background:#fff!important;color:#17334b!important;border:1px solid #a8bccb!important;border-radius:9px}
    .block-container button[kind="secondary"] *,.block-container [data-testid="stBaseButton-secondary"] *{color:#17334b!important}
    .block-container button[kind="primary"]{background:#087f8c!important;border-color:#087f8c!important;border-radius:9px}
    .block-container button[kind="primary"] *{color:white!important}
    .block-container [role="tablist"] button{color:#375368!important;font-weight:600}
    .block-container [role="tablist"] button[aria-selected="true"]{color:#087f8c!important}
    [data-testid="stFileUploader"]{background:#fff;border:1px dashed #a8bccb;border-radius:14px;padding:8px}
    .block-container [data-testid="stFileUploaderDropzone"],.block-container [data-testid="stFileUploaderFile"]{background:white!important;color:#17334b!important}
    [data-testid="stChatMessage"]{background:#fff;border:1px solid #d7e2eb;border-radius:14px}
    .block-container [data-testid="stChatInput"]{background:#fff!important;border:1px solid #a8bccb;border-radius:12px}
    .block-container [data-testid="stChatInput"] textarea{background:#fff!important;color:#17334b!important;caret-color:#17334b}
    .block-container [data-testid="stChatInput"] textarea::placeholder{color:#536d80!important;opacity:1}
    @media(max-width:800px){.research-hero{grid-template-columns:1fr;padding:25px}.signal-art{display:none}.research-hero h1{font-size:2rem}}
    </style>''',unsafe_allow_html=True)
