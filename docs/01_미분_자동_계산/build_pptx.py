#!/usr/bin/env python3
"""build_pptx.py

Converts slides.md into a high-quality widescreen (16:9) PowerPoint presentation (01_미분_자동_계산.pptx)
in strict accordance with PPTX_CONVERSION_GUIDE.md and SLIDES_CONVERSION_GUIDE.md.
"""

import os
import re
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR

# ==========================================
# 1. 디자인 상수 정의 (16:9 규격 및 컬러 시스템)
# ==========================================
SLIDE_WIDTH = Inches(13.333)
SLIDE_HEIGHT = Inches(7.500)

COLOR_PRIMARY = RGBColor(26, 115, 232)      # #1A73E8 (Brand Blue Badge)
COLOR_TITLE = RGBColor(17, 24, 39)          # #111827 (Dark Charcoal, 거의 검정)
COLOR_SUBTITLE = RGBColor(107, 114, 128)    # #6B7280 (Medium Gray, 회색)
COLOR_BODY = RGBColor(31, 41, 55)           # #1F2937 (Body Text)
COLOR_DIVIDER = RGBColor(226, 232, 240)     # #E2E8F0 (Divider Line, 깔끔한 연회색)
COLOR_CAPTION = RGBColor(75, 85, 99)        # #4B5563 (Caption Gray)
COLOR_WHITE = RGBColor(255, 255, 255)       # #FFFFFF (Pure White)
COLOR_DARK_TEXT = RGBColor(30, 41, 59)      # #1E293B

FONT_NAME = "나눔고딕"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SLIDES_MD_PATH = os.path.join(BASE_DIR, "slides.md")
COVER_PNG_PATH = os.path.join(BASE_DIR, "01_미분_자동_계산.png")
OUTPUT_PPTX_PATH = os.path.join(BASE_DIR, "01_미분_자동_계산.pptx")

# ==========================================
# 2. 텍스트 및 수식 정제 함수 (LaTeX / HTML / Markdown -> Clean Unicode)
# ==========================================
SUP_MAP = {
    '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
    '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
    '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾',
    'n': 'ⁿ', 'i': 'ⁱ', 'T': 'ᵀ', 'N': 'ᴺ', 'x': 'ˣ', 'y': 'ʸ'
}
SUB_MAP = {
    '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
    '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
    '+': '₊', '-': '₋', '=': '₌', '(': '₍', ')': '₎',
    'a': 'ₐ', 'e': 'ₑ', 'h': 'ₕ', 'i': 'ᵢ', 'j': 'ⱼ',
    'k': 'ₖ', 'l': 'ₗ', 'm': 'ₘ', 'n': 'ₙ', 'o': 'ₒ',
    'p': 'ₚ', 'r': 'ᵣ', 's': 'ₛ', 't': 'ₜ', 'u': 'ᵤ',
    'v': 'ᵥ', 'x': 'ₓ'
}

def clean_slide_math(text: str) -> str:
    """슬라이드 본문용 클린 유니코드 변환"""
    if not text:
        return ""
    
    # 1. HTML 태그 변환
    def sup_repl(m):
        content = m.group(1).replace('*', '')
        return "".join(SUP_MAP.get(c, c) for c in content)
    def sub_repl(m):
        content = m.group(1).replace('*', '')
        return "".join(SUB_MAP.get(c, c) for c in content)

    text = re.sub(r"<sup>(.*?)</sup>", sup_repl, text, flags=re.IGNORECASE)
    text = re.sub(r"<sub>(.*?)</sub>", sub_repl, text, flags=re.IGNORECASE)
    text = re.sub(r"<div.*?>.*?</div>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?(b|strong|i|em|code|span|p|br|div)>", "", text, flags=re.IGNORECASE)
    
    # 2. 마크다운 링크 [텍스트](url) -> 텍스트
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    
    # 3. 마크다운 인라인 서식 (`code`, **bold**, *italic*)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    
    # 4. LaTeX 그리스 문자 및 연산자 변환
    greek_ops = [
        (r"\\alpha", "α"), (r"\\beta", "β"), (r"\\gamma", "γ"),
        (r"\\delta", "δ"), (r"\\epsilon|\\varepsilon", "ε"),
        (r"\\theta", "θ"), (r"\\lambda", "λ"), (r"\\mu", "μ"),
        (r"\\pi", "π"), (r"\\rho", "ρ"), (r"\\sigma", "σ"),
        (r"\\tau", "τ"), (r"\\nabla", "∇"), (r"\\partial", "∂"),
        (r"\\times", "×"), (r"\\cdot", "·"), (r"\\rightarrow|\\to", "→"),
        (r"\\leftarrow", "←"), (r"\\approx", "≈"), (r"\\le|\\leq", "≤"),
        (r"\\ge|\\geq", "≥"), (r"\\neq", "≠"), (r"\\sum", "∑")
    ]
    for pat, rep in greek_ops:
        text = re.sub(pat, rep, text)
        
    # 5. 수식 구분자 제거 ($...$, $$...$$)
    text = text.replace("$$", "").replace("$", "")
    
    # 6. 잔여 백슬래시 및 원화 기호 제거
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = text.replace("\\", "").replace("₩", "")
    
    return text.strip()

def clean_notes_math(text: str) -> str:
    """슬라이드 노트용 TTS 발음 최적화 구술문 정제"""
    if not text:
        return ""
    # 마크다운 서식 정리
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"</?[^>]+>", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = text.replace("$$", "").replace("$", "")
    text = text.replace("\\", "").replace("₩", "")
    return text.strip()

# ==========================================
# 3. PPTX 생성 클래스
# ==========================================
class DezeroPresentationBuilder:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width = SLIDE_WIDTH
        self.prs.slide_height = SLIDE_HEIGHT
        self.blank_layout = self.prs.slide_layouts[6]

    def create_cover_slide(self, badge_text, title_text, subtitle_lines, script_text=""):
        """표지 슬라이드: 우하단 84.5% 커버 이미지 + 좌상단 3단 요소 (배지, 메인 타이틀, 서브 설명문)"""
        slide = self.prs.slides.add_slide(self.blank_layout)

        # 1. 하위 레이어: 커버 이미지 우하단 고정 배치 (84.5% 축소)
        if os.path.exists(COVER_PNG_PATH):
            img_w = Inches(11.27)
            img_h = Inches(6.34)
            left = Inches(2.06)
            top = Inches(1.16)
            slide.shapes.add_picture(COVER_PNG_PATH, left, top, width=img_w, height=img_h)

        # 2. 상위 레이어 1: 챕터 정보 배지 (x=0.8", y=0.60", w=2.6", h=0.42", 13pt Bold)
        badge = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.60), Inches(2.6), Inches(0.42)
        )
        badge.fill.solid()
        badge.fill.fore_color.rgb = COLOR_PRIMARY
        badge.line.color.rgb = COLOR_PRIMARY
        badge.adjustments[0] = 0.5

        tf = badge.text_frame
        tf.word_wrap = False
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = badge_text
        p.font.name = FONT_NAME
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = COLOR_WHITE
        p.alignment = PP_ALIGN.CENTER

        # 3. 상위 레이어 2: 메인 타이틀 (x=0.8", y=1.22", w=7.0", h=0.65", 38pt Bold)
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.22), Inches(7.0), Inches(0.65))
        tf = title_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.name = FONT_NAME
        p.font.size = Pt(38)
        p.font.bold = True
        p.font.color.rgb = COLOR_TITLE

        # 4. 상위 레이어 3: 서브 설명문 (x=0.8", y=1.98", w=7.5", h=1.10", 19pt Regular, #4B5563)
        sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.98), Inches(7.5), Inches(1.10))
        tf = sub_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        for i, line in enumerate(subtitle_lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line
            p.font.name = FONT_NAME
            p.font.size = Pt(19)
            p.font.color.rgb = COLOR_CAPTION
            p.line_spacing = 1.35

        # 5. 슬라이드 노트 (TTS 최적화 구술문)
        if script_text:
            notes_slide = slide.notes_slide
            notes_slide.notes_text_frame.text = clean_notes_math(script_text)

    def add_common_header(self, slide, title_text, subtitle_text, page_str):
        """본문 슬라이드 고정 헤더: 제목(32pt), 부제(20pt), 페이지번호(14pt), 구분선(1px)"""
        # 제목: x 0.8", y 0.4", w 10.5", h 0.55" / 32pt Bold
        t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(10.5), Inches(0.55))
        tf = t_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = clean_slide_math(title_text)
        p.font.name = FONT_NAME
        p.font.size = Pt(32)
        p.font.bold = True
        p.font.color.rgb = COLOR_TITLE

        # 부제: x 0.8", y 1.05", w 10.5", h 0.45" / 20pt Regular
        s_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.05), Inches(10.5), Inches(0.45))
        sf = s_box.text_frame
        sf.word_wrap = True
        sf.margin_left = sf.margin_right = sf.margin_top = sf.margin_bottom = 0
        sp = sf.paragraphs[0]
        sp.text = clean_slide_math(subtitle_text)
        sp.font.name = FONT_NAME
        sp.font.size = Pt(20)
        sp.font.color.rgb = COLOR_SUBTITLE

        # 페이지 번호: x 11.33", y 0.5", w 1.2", h 0.4" / 14pt Bold
        num_box = slide.shapes.add_textbox(Inches(11.333), Inches(0.5), Inches(1.2), Inches(0.4))
        np_f = num_box.text_frame
        np_f.margin_left = np_f.margin_right = np_f.margin_top = np_f.margin_bottom = 0
        np_p = np_f.paragraphs[0]
        np_p.text = page_str
        np_p.font.name = FONT_NAME
        np_p.font.size = Pt(14)
        np_p.font.bold = True
        np_p.font.color.rgb = RGBColor(156, 163, 175)
        np_p.alignment = PP_ALIGN.RIGHT

        # 구분선: y 1.7", x 0.8" ~ 12.533" (길이 11.733"), #E2E8F0, 1px (Pt(0.75))
        line = slide.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT, Inches(0.8), Inches(1.7), Inches(12.533), Inches(1.7)
        )
        line.line.color.rgb = COLOR_DIVIDER
        line.line.width = Pt(0.75)

    def create_2column_image_slide(self, title_text, subtitle_text, image_path, caption_text, body_paragraphs, page_str, script_text=""):
        """2단 이미지 슬라이드: 좌측 이미지(4.2x3.9 fit) + 캡션(13pt), 우측 텍스트(22pt, 세로 중앙 정렬)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.add_common_header(slide, title_text, subtitle_text, page_str)

        # 좌측 이미지 (x=1.0", y=2.15", max_w=4.2", max_h=3.9" 비율 유지)
        if image_path and os.path.exists(image_path):
            max_w = Inches(4.2)
            max_h = Inches(3.9)
            with Image.open(image_path) as img:
                orig_w, orig_h = img.size
            scale = min(max_w / orig_w, max_h / orig_h)
            dest_w = int(orig_w * scale)
            dest_h = int(orig_h * scale)
            dest_left = Inches(1.0) + int((max_w - dest_w) / 2)
            dest_top = Inches(2.15) + int((max_h - dest_h) / 2)

            slide.shapes.add_picture(image_path, dest_left, dest_top, width=dest_w, height=dest_h)

            # 캡션 (이미지 아래 y = dest_top + dest_h + 0.18", 13pt Regular, 가운데 정렬)
            if caption_text:
                cap_top = dest_top + dest_h + Inches(0.18)
                c_box = slide.shapes.add_textbox(dest_left, cap_top, dest_w, Inches(0.6))
                cf = c_box.text_frame
                cf.word_wrap = True
                cf.margin_left = cf.margin_right = cf.margin_top = cf.margin_bottom = 0
                cp = cf.paragraphs[0]
                cp.text = clean_slide_math(caption_text)
                cp.font.name = FONT_NAME
                cp.font.size = Pt(13)
                cp.font.color.rgb = COLOR_CAPTION
                cp.alignment = PP_ALIGN.CENTER

        # 우측 본문 텍스트 영역 (x=6.0", y=1.90", w=6.4", h=4.8", 세로 중앙 정렬)
        t_box = slide.shapes.add_textbox(Inches(6.0), Inches(1.90), Inches(6.4), Inches(4.8))
        tf = t_box.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

        for i, text in enumerate(body_paragraphs):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            cleaned = clean_slide_math(text)
            p.text = f"• {cleaned}"
            p.font.name = FONT_NAME
            p.font.size = Pt(21)
            p.font.color.rgb = COLOR_BODY
            p.line_spacing = 1.45
            p.space_after = Pt(20)

        if script_text:
            notes_slide = slide.notes_slide
            notes_slide.notes_text_frame.text = clean_notes_math(script_text)

    def create_wide_text_slide(self, title_text, subtitle_text, paragraphs, page_str, script_text=""):
        """와이드 텍스트 슬라이드 (x=0.8", y=1.90", w=11.733", h=4.8", 세로 중앙 정렬)"""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self.add_common_header(slide, title_text, subtitle_text, page_str)

        t_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.90), Inches(11.733), Inches(4.8))
        tf = t_box.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

        # 분량에 따른 폰트 크기 동적 조절
        total_chars = sum(len(p) for p in paragraphs)
        if len(paragraphs) > 4 or total_chars > 350:
            font_size = Pt(17)
            line_spacing = 1.35
            space_after = Pt(12)
        elif len(paragraphs) > 3 or total_chars > 250:
            font_size = Pt(19)
            line_spacing = 1.40
            space_after = Pt(16)
        else:
            font_size = Pt(21)
            line_spacing = 1.45
            space_after = Pt(20)

        for i, text in enumerate(paragraphs):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            cleaned = clean_slide_math(text)
            # 글머리 기호가 이미 포함되어 있지 않은 경우 적절히 부착
            if not re.match(r"^(\d+\.|\•|\-)", cleaned):
                p.text = f"• {cleaned}"
            else:
                p.text = cleaned
            p.font.name = FONT_NAME
            p.font.size = font_size
            p.font.color.rgb = COLOR_BODY
            p.line_spacing = line_spacing
            p.space_after = space_after

        if script_text:
            notes_slide = slide.notes_slide
            notes_slide.notes_text_frame.text = clean_notes_math(script_text)

    def create_closing_slide(self, badge_text, title_text, subtitle_lines, qa_text, page_str, script_text=""):
        """종료 슬라이드: 완강 배지, 감사합니다(46pt Bold), 요약 설명, Q&A 및 토론"""
        slide = self.prs.slides.add_slide(self.blank_layout)

        # 페이지 번호: x 11.33", y 0.5"
        num_box = slide.shapes.add_textbox(Inches(11.333), Inches(0.5), Inches(1.2), Inches(0.4))
        np_f = num_box.text_frame
        np_f.margin_left = np_f.margin_right = np_f.margin_top = np_f.margin_bottom = 0
        np_p = np_f.paragraphs[0]
        np_p.text = page_str
        np_p.font.name = FONT_NAME
        np_p.font.size = Pt(14)
        np_p.font.bold = True
        np_p.font.color.rgb = RGBColor(156, 163, 175)
        np_p.alignment = PP_ALIGN.RIGHT

        # 1. 완강 배지: x=5.166", y=2.2", w=3.0", h=0.44", 알약 모양
        badge = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(5.166), Inches(2.2), Inches(3.0), Inches(0.44)
        )
        badge.fill.solid()
        badge.fill.fore_color.rgb = COLOR_PRIMARY
        badge.line.color.rgb = COLOR_PRIMARY
        badge.adjustments[0] = 0.5

        tf = badge.text_frame
        tf.word_wrap = False
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = badge_text
        p.font.name = FONT_NAME
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = COLOR_WHITE
        p.alignment = PP_ALIGN.CENTER

        # 2. 메인 감사 메시지: x=2.0", y=2.9", w=9.333", h=1.0", 46pt Bold, #1E293B, Center
        title_box = slide.shapes.add_textbox(Inches(2.0), Inches(2.9), Inches(9.333), Inches(1.0))
        tf = title_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.name = FONT_NAME
        p.font.size = Pt(46)
        p.font.bold = True
        p.font.color.rgb = COLOR_DARK_TEXT
        p.alignment = PP_ALIGN.CENTER

        # 3. 요약 및 다음 단계 안내: x=2.0", y=4.05", w=9.333", h=0.8", 18pt Regular, #4B5563, Center
        sub_box = slide.shapes.add_textbox(Inches(2.0), Inches(4.05), Inches(9.333), Inches(0.8))
        tf = sub_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        for i, line in enumerate(subtitle_lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line
            p.font.name = FONT_NAME
            p.font.size = Pt(18)
            p.font.color.rgb = COLOR_CAPTION
            p.line_spacing = 1.40
            p.alignment = PP_ALIGN.CENTER

        # 4. Q&A 안내: x=2.0", y=5.1", w=9.333", h=0.45", 15pt Bold, #1A73E8, Center
        qa_box = slide.shapes.add_textbox(Inches(2.0), Inches(5.1), Inches(9.333), Inches(0.45))
        tf = qa_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = qa_text
        p.font.name = FONT_NAME
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY
        p.alignment = PP_ALIGN.CENTER

        if script_text:
            notes_slide = slide.notes_slide
            notes_slide.notes_text_frame.text = clean_notes_math(script_text)


# ==========================================
# 4. slides.md 파서 및 빌드 메인 루틴
# ==========================================
def parse_slides_md(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # '---' 구분자로 슬라이드 분할
    raw_slides = re.split(r"\n---\s*\n", content)
    slides_data = []

    for raw in raw_slides:
        raw = raw.strip()
        if not raw:
            continue

        lines = raw.split("\n")
        title = ""
        subtitle = ""
        image_path = ""
        caption = ""
        script = ""
        body_lines = []

        # 스크립트 분리 (> 블록)
        script_parts = []
        non_script_lines = []
        for line in lines:
            if line.strip().startswith(">"):
                script_parts.append(line.strip().lstrip(">").strip())
            else:
                non_script_lines.append(line)
        script = " ".join(script_parts)

        # 제목, 부제, 이미지, 캡션, 본문 파싱
        for line in non_script_lines:
            line_str = line.strip()
            if not line_str:
                continue

            if line_str.startswith("# ") and not title:
                title = line_str.lstrip("#").strip()
            elif line_str.startswith("## ") and not subtitle:
                subtitle = line_str.lstrip("#").strip()
            elif re.match(r"^!\[.*?\]\((.*?)\)", line_str):
                m = re.match(r"^!\[.*?\]\((.*?)\)", line_str)
                rel_path = m.group(1).strip()
                if rel_path.startswith("./"):
                    rel_path = rel_path[2:]
                image_path = os.path.join(BASE_DIR, rel_path)
            elif re.match(r"^\*\*그림\s*[\d\w\-]+\*\*", line_str):
                caption = line_str
            else:
                body_lines.append(line_str)

        slides_data.append({
            "title": title,
            "subtitle": subtitle,
            "image": image_path,
            "caption": caption,
            "body": body_lines,
            "script": script
        })

    return slides_data


def main():
    print(f"[*] Reading slides from: {SLIDES_MD_PATH}")
    parsed_slides = parse_slides_md(SLIDES_MD_PATH)
    print(f"[*] Parsed {len(parsed_slides)} slide sections from slides.md")

    builder = DezeroPresentationBuilder()

    # 총 슬라이드 수 = 1(Cover) + len(parsed_slides) (참고: parsed_slides의 마지막은 closing slide)
    # 총 슬라이드 9장:
    # 01: Cover Slide
    # 02: 01.0 무에서 시작하여 자동 미분의 엔진을 완성하다 (Image 2-column)
    # 03: 01.1 학습 목표 (Wide Text)
    # 04: 01.2 학습 목차 (1/3)
    # 05: 01.3 학습 목차 (2/3)
    # 06: 01.4 학습 목차 (3/3)
    # 07: 01.5 핵심 학습 요약 (1/2)
    # 08: 01.6 핵심 학습 요약 (2/2)
    # 09: 01.7 제1고지 완강 / 감사 (Closing Slide)
    total_slides = len(parsed_slides) + 1

    # 1. Slide 1: Cover Slide
    cover_subtitle_lines = [
        "단 3줄의 변수 상자에서 출발하여 계산 그래프와",
        "자동 역전파(Backpropagation) 엔진을 완성하는 여정"
    ]
    cover_script = (
        "제1고지 미분 자동 계산에 오신 여러분을 환영합니다. "
        "우리는 라이브러리의 블랙박스를 걷어내고, 단 세 줄의 변수 상자 정의에서 출발하여 "
        "스스로 미분을 수행하는 역전파 엔진을 한 땀 한 땀 조립해 나갈 것입니다. "
        "지니와 도로시 탐험대와 함께 기초 수학과 코드로 자동 미분의 본질을 완벽히 정복해 봅시다."
    )
    print("[-] Generating Slide 01: Cover Slide")
    builder.create_cover_slide(
        badge_text="PART 01 · 미분 자동 계산",
        title_text="제1고지: 미분 자동 계산",
        subtitle_lines=cover_subtitle_lines,
        script_text=cover_script
    )

    # 2. Body & Closing Slides (parsed_slides 0 through 7 -> Slides 02 through 09)
    for idx, slide_info in enumerate(parsed_slides):
        page_num = idx + 2
        page_str = f"{page_num:02d} / {total_slides:02d}"
        
        # 마지막 슬라이드는 Closing Slide로 렌더링
        if idx == len(parsed_slides) - 1:
            print(f"[-] Generating Slide {page_num:02d}: Closing Slide")
            closing_sub_lines = [
                "단 3줄의 Variable 상자에서 출발하여 계산 그래프와 자동 역전파 엔진의 기초를 완성했습니다.",
                "다음 제2고지에서는 다중 입출력과 자연스러운 연산자 오버로딩의 세계로 나아갑니다."
            ]
            builder.create_closing_slide(
                badge_text="PART 01 · 미분 자동 계산 완강",
                title_text="감사합니다",
                subtitle_lines=closing_sub_lines,
                qa_text="질의응답(Q&A) 및 토론",
                page_str=page_str,
                script_text=slide_info["script"]
            )
        # 이미지가 있는 슬라이드는 2단 이미지 슬라이드로 렌더링
        elif slide_info["image"]:
            print(f"[-] Generating Slide {page_num:02d}: 2-Column Image Slide ({slide_info['subtitle']})")
            builder.create_2column_image_slide(
                title_text=slide_info["title"],
                subtitle_text=slide_info["subtitle"],
                image_path=slide_info["image"],
                caption_text=slide_info["caption"],
                body_paragraphs=slide_info["body"],
                page_str=page_str,
                script_text=slide_info["script"]
            )
        # 일반 텍스트 슬라이드는 와이드 텍스트 슬라이드로 렌더링
        else:
            print(f"[-] Generating Slide {page_num:02d}: Wide Text Slide ({slide_info['subtitle']})")
            builder.create_wide_text_slide(
                title_text=slide_info["title"],
                subtitle_text=slide_info["subtitle"],
                paragraphs=slide_info["body"],
                page_str=page_str,
                script_text=slide_info["script"]
            )

    print(f"[*] Saving presentation to: {OUTPUT_PPTX_PATH}")
    builder.prs.save(OUTPUT_PPTX_PATH)
    print("[+] Successfully created PPTX presentation!")


if __name__ == "__main__":
    main()
