# ═══════════════════════════════════════════════════════
#  app.py  —  واجهة Streamlit
#  تشغيل: streamlit run app.py
# ═══════════════════════════════════════════════════════
import streamlit as st
from sympy import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from solver import *

# ── إعداد الصفحة ────────────────────────────────────────
st.set_page_config(
    page_title="حل المعادلات التفاضلية",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS مخصص ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700;900&display=swap');

* { font-family: 'Cairo', sans-serif !important; }

.main { background: #07071a; }

.stApp {
    background: linear-gradient(135deg, #07071a 0%, #0b0b22 100%);
    direction: rtl;
}

h1, h2, h3 { color: #00d4ff !important; }

.step-card {
    background: rgba(0,212,255,0.05);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 10px 0;
    direction: rtl;
}

.step-num {
    display: inline-block;
    background: linear-gradient(135deg, #00d4ff, #b44fff);
    color: #000;
    font-weight: 900;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    text-align: center;
    line-height: 32px;
    margin-left: 10px;
    font-size: 14px;
}

.result-box {
    background: rgba(0,255,136,0.08);
    border: 2px solid rgba(0,255,136,0.4);
    border-radius: 14px;
    padding: 20px;
    margin: 16px 0;
    text-align: center;
    direction: ltr;
}

.result-label {
    color: #00ff88;
    font-weight: 900;
    font-size: 16px;
    margin-bottom: 8px;
    text-align: right;
    direction: rtl;
}

.tip-box {
    background: rgba(255,215,0,0.07);
    border: 1px solid rgba(255,215,0,0.25);
    border-radius: 10px;
    padding: 12px 16px;
    color: #ffd700;
    direction: rtl;
}

.header-badge {
    background: linear-gradient(135deg,rgba(0,212,255,.1),rgba(180,79,255,.1));
    border: 1px solid rgba(0,212,255,.3);
    border-radius: 20px;
    padding: 6px 20px;
    color: #00d4ff;
    font-size: 13px;
    letter-spacing: 2px;
    display: inline-block;
    margin-bottom: 10px;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stNumberInput"] label {
    color: rgba(255,255,255,0.8) !important;
    font-size: 14px !important;
}

.stSelectbox > div > div,
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: white !important;
    border-radius: 10px !important;
    direction: ltr !important;
}

.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #b44fff) !important;
    color: #000 !important;
    font-weight: 900 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 30px !important;
    font-size: 16px !important;
    width: 100% !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,212,255,0.3) !important;
}

hr { border-color: rgba(255,255,255,0.08) !important; }

.sidebar .sidebar-content {
    background: rgba(0,0,0,0.4) !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════
st.markdown('<div style="text-align:center">', unsafe_allow_html=True)
st.markdown('<div class="header-badge">∂ &nbsp; ODE SOLVER &nbsp; ∂</div>', unsafe_allow_html=True)
st.markdown("""
<h1 style="font-size:2.8rem; font-weight:900; margin:0;
    background:linear-gradient(135deg,#00d4ff,#b44fff,#ff6b9d);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
    حل المعادلات التفاضلية
</h1>
<p style="color:rgba(255,255,255,0.45); font-size:15px; margin-top:8px;">
    تصنيف تلقائي · خطوة بخطوة · رسم بياني
</p>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")


# ══════════════════════════════════════════════════════════
#  SIDEBAR — الإعدادات
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ إعدادات الحل")
    st.markdown("---")

    order = st.selectbox(
        "رتبة المعادلة",
        ["الرتبة الأولى — First Order",
         "الرتبة الثانية — Second Order",
         "تلقائي — Auto Detect"]
    )

    st.markdown("---")

    if "الأولى" in order:
        method = st.selectbox(
            "طريقة الحل",
            ["تلقائي (SymPy)",
             "فصل المتغيرات",
             "خطية (Linear)",
             "برنولي (Bernoulli)",
             "تامة (Exact)",
             "متجانسة (Homogeneous)"]
        )
    elif "الثانية" in order:
        method = st.selectbox(
            "نوع المعادلة",
            ["تلقائي (SymPy)",
             "معاملات ثابتة (Constant Coefficients)",
             "كوشي-أويلر (Cauchy-Euler)",
             "غير متجانسة (Non-Homogeneous)"]
        )
    else:
        method = "تلقائي (SymPy)"

    st.markdown("---")
    show_plot = st.checkbox("📊 رسم الحل بيانياً", value=True)
    show_verify = st.checkbox("🔍 التحقق من الحل", value=True)

    st.markdown("---")
    st.markdown("""
    <div style="color:rgba(255,255,255,0.4); font-size:12px; text-align:center">
    📖 كيفية إدخال المعادلة:<br>
    <code style="color:#00d4ff">y(x)</code> للمجهول<br>
    <code style="color:#00d4ff">y(x).diff(x)</code> للمشتق الأول<br>
    <code style="color:#00d4ff">y(x).diff(x,2)</code> للمشتق الثاني<br>
    <code style="color:#00d4ff">exp(x), sin(x), cos(x)</code><br>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  المحتوى الرئيسي
# ══════════════════════════════════════════════════════════
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 📝 أدخل المعادلة")

    # ── حقل إدخال المعادلة ──
    x_sym = symbols('x')
    y_fn  = Function('y')

    if "تلقائي" in method or "الثانية" in order:
        eq_input = st.text_input(
            "المعادلة (صيغة SymPy)",
            value="y(x).diff(x) + 2*y(x) - 4*x",
            help="مثال: y(x).diff(x,2) - 3*y(x).diff(x) + 2*y(x)"
        )
        st.caption("💡 الصيغة: `LHS - RHS = 0`  أو  `y(x).diff(x) + ...`")

    elif method == "فصل المتغيرات":
        st.markdown("**الشكل: dy/dx = f(x) · g(y)**")
        c1, c2 = st.columns(2)
        with c1:
            fx_input = st.text_input("f(x) =", value="x", help="دالة في x فقط")
        with c2:
            gy_input = st.text_input("g(y) =", value="y**2", help="دالة في y فقط")

    elif method == "خطية (Linear)":
        st.markdown("**الشكل: dy/dx + P(x)·y = Q(x)**")
        c1, c2 = st.columns(2)
        with c1:
            px_input = st.text_input("P(x) =", value="2/x")
        with c2:
            qx_input = st.text_input("Q(x) =", value="x**2")

    elif method == "برنولي (Bernoulli)":
        st.markdown("**الشكل: dy/dx + P(x)·y = Q(x)·yⁿ**")
        c1, c2, c3 = st.columns(3)
        with c1:
            px_input = st.text_input("P(x) =", value="-1")
        with c2:
            qx_input = st.text_input("Q(x) =", value="x")
        with c3:
            n_input  = st.text_input("n =", value="3")

    elif method == "تامة (Exact)":
        st.markdown("**الشكل: M(x,y)dx + N(x,y)dy = 0**")
        c1, c2 = st.columns(2)
        with c1:
            m_input = st.text_input("M(x,y) =", value="2*x*y + y**2")
        with c2:
            n_input_e = st.text_input("N(x,y) =", value="x**2 + 2*x*y")

    elif method == "معاملات ثابتة (Constant Coefficients)":
        st.markdown("**الشكل: ay'' + by' + cy = 0**")
        c1, c2, c3 = st.columns(3)
        with c1:
            a_input = st.number_input("a =", value=1.0)
        with c2:
            b_input = st.number_input("b =", value=-3.0)
        with c3:
            c_input = st.number_input("c =", value=2.0)

    elif method == "كوشي-أويلر (Cauchy-Euler)":
        st.markdown("**الشكل: ax²y'' + bxy' + cy = 0**")
        c1, c2, c3 = st.columns(3)
        with c1:
            a_input = st.number_input("a =", value=1.0)
        with c2:
            b_input = st.number_input("b =", value=-3.0)
        with c3:
            c_input = st.number_input("c =", value=4.0)

    # ── شروط ابتدائية ──
    st.markdown("---")
    st.markdown("### 🎯 شروط ابتدائية (اختياري)")
    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        x0_val = st.number_input("x₀ =", value=0.0)
    with ic2:
        y0_val = st.text_input("y(x₀) =", value="", placeholder="مثال: 1")
    with ic3:
        dy0_val = st.text_input("y'(x₀) =", value="", placeholder="للرتبة الثانية")

    st.markdown("---")
    solve_btn = st.button("⚡ احسب الحل الآن", use_container_width=True)


# ══════════════════════════════════════════════════════════
#  منطقة النتائج
# ══════════════════════════════════════════════════════════
with col2:
    st.markdown("### 📊 أمثلة سريعة")
    examples = {
        "y' + 2y = 4x":      "y(x).diff(x) + 2*y(x) - 4*x",
        "y' = x·y²":         "y(x).diff(x) - x*y(x)**2",
        "y'' - 3y' + 2y = 0":"y(x).diff(x,2) - 3*y(x).diff(x) + 2*y(x)",
        "y'' + 4y = 0":      "y(x).diff(x,2) + 4*y(x)",
        "x²y'' - 3xy' + 4y = 0": "x**2*y(x).diff(x,2) - 3*x*y(x).diff(x) + 4*y(x)",
    }
    for name, expr in examples.items():
        if st.button(f"📌 {name}", use_container_width=True):
            st.session_state['example_eq'] = expr
            st.rerun()

    if 'example_eq' in st.session_state:
        st.info(f"✅ تم نسخ: `{st.session_state['example_eq']}`\nالصقيه في خانة المعادلة!")


# ══════════════════════════════════════════════════════════
#  تنفيذ الحل
# ══════════════════════════════════════════════════════════
if solve_btn:
    st.markdown("---")
    st.markdown("## 📐 نتيجة الحل")

    x_s = symbols('x')
    y_f = Function('y')

    with st.spinner("⏳ جاري الحل..."):
        try:
            steps   = []
            sol_str = ""
            error   = None

            # ── تحضير الشروط الابتدائية ──
            ics = {}
            if y0_val.strip():
                ics[y_f(x0_val)] = sympify(y0_val)
            if dy0_val.strip():
                ics[y_f(x_s).diff(x_s).subs(x_s, x0_val)] = sympify(dy0_val)

            # ── اختيار الطريقة ──
            if method == "فصل المتغيرات":
                steps, sol_str = solve_separation(fx_input, gy_input)

            elif method == "خطية (Linear)":
                steps, sol_str = solve_linear_first(px_input, qx_input)

            elif method == "برنولي (Bernoulli)":
                steps, sol_str = solve_bernoulli_method(px_input, qx_input, n_input)

            elif method == "تامة (Exact)":
                steps, sol_str = solve_exact_method(m_input, n_input_e)

            elif method == "معاملات ثابتة (Constant Coefficients)":
                steps, sol_str = solve_const_coeff(int(a_input), int(b_input), int(c_input))

            elif method == "كوشي-أويلر (Cauchy-Euler)":
                steps, sol_str = solve_cauchy_euler_method(int(a_input), int(b_input), int(c_input))

            else:
                # تلقائي
                ode_eq = Eq(sympify(eq_input), 0)

                # التصنيف
                try:
                    ord_n, cls_list = get_order_and_type(ode_eq)
                    st.markdown(f"""
                    <div class="step-card">
                        <b style="color:#00d4ff">🔍 التصنيف التلقائي</b><br>
                        الرتبة: <b style="color:#00ff88">{ord_n}</b><br>
                        النوع: <code style="color:#ff9500">{cls_list[0] if cls_list else 'غير محدد'}</code>
                    </div>
                    """, unsafe_allow_html=True)
                except:
                    pass

                sol_obj, error = auto_solve_ode(ode_eq, ics if ics else None)
                if sol_obj:
                    sol_str = str(sol_obj.rhs)
                    steps = [("الحل بـ SymPy", str(sol_obj), "")]

            # ── عرض الخطوات ──
            if steps:
                st.markdown("### 📋 خطوات الحل")
                for i, step in enumerate(steps):
                    title, work, detail = step
                    if "خطأ" in title:
                        st.error(f"⚠️ {work}")
                        continue
                    st.markdown(f"""
                    <div class="step-card">
                        <span class="step-num">{i+1}</span>
                        <b style="color:white; font-size:15px">{title}</b>
                        <div style="background:rgba(0,0,0,0.3); border-radius:8px;
                            padding:10px 14px; margin-top:10px; font-family:monospace;
                            color:#00d4ff; direction:ltr; text-align:left; font-size:14px;">
                            {work}
                        </div>
                        {"<div style='color:rgba(255,255,255,0.5);font-size:12px;margin-top:6px;'>"+detail+"</div>" if detail else ""}
                    </div>
                    """, unsafe_allow_html=True)

            # ── عرض الحل النهائي ──
            if sol_str and "خطأ" not in sol_str:
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-label">✅ الحل العام</div>
                    <div style="font-size:20px; font-weight:700; color:#00ff88; font-family:monospace;">
                        y = {sol_str}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            elif error:
                st.error(f"⚠️ لم يتمكن من الحل: {error}")

            # ── التحقق ──
            if show_verify and sol_str and "تلقائي" in method:
                st.markdown("""
                <div class="tip-box">
                💡 <b>للتحقق:</b> عوّض الحل في المعادلة الأصلية — يجب أن تتحقق المعادلة
                </div>
                """, unsafe_allow_html=True)

            # ── الرسم البياني ──
            if show_plot and sol_str and sol_str != "":
                st.markdown("### 📈 الرسم البياني")
                try:
                    C1, C2 = symbols('C1 C2')
                    sol_expr = sympify(sol_str)

                    fig, ax = plt.subplots(figsize=(9, 4))
                    ax.set_facecolor('#0d0d1a')
                    fig.patch.set_facecolor('#07071a')

                    x_vals  = np.linspace(-3, 3, 500)
                    colors  = ['#00d4ff','#00ff88','#ff9500','#ff6b9d','#b44fff']
                    c_vals  = [-2, -1, 0, 1, 2]
                    plotted = 0

                    for i, cv in enumerate(c_vals):
                        try:
                            expr = sol_expr
                            if C1 in expr.free_symbols:
                                expr = expr.subs(C1, cv)
                            if C2 in expr.free_symbols:
                                expr = expr.subs(C2, 0)
                            f      = lambdify(x_s, expr, 'numpy')
                            y_vals = np.array(f(x_vals), dtype=float)
                            y_vals = np.where(np.abs(y_vals) > 20, np.nan, y_vals)
                            if not np.all(np.isnan(y_vals)):
                                ax.plot(x_vals, y_vals, color=colors[i],
                                        linewidth=2.2, alpha=0.85,
                                        label=f'C = {cv}')
                                plotted += 1
                        except:
                            pass

                    if plotted > 0:
                        ax.axhline(0, color='rgba(255,255,255,0.15)', lw=0.8)
                        ax.axvline(0, color='rgba(255,255,255,0.15)', lw=0.8)
                        ax.set_xlabel('x', color='white')
                        ax.set_ylabel('y', color='white')
                        ax.set_title('عائلة الحلول', color='white', pad=10)
                        ax.tick_params(colors='white')
                        ax.spines[:].set_color('#2a2a4a')
                        ax.legend(facecolor='#0d0d2a', labelcolor='white', fontsize=9)
                        ax.set_ylim(-10, 10)
                        ax.grid(True, alpha=0.1, color='#4444aa')
                        st.pyplot(fig)
                    else:
                        st.info("📊 لا يمكن رسم هذا الحل تلقائياً")
                    plt.close()
                except Exception as e:
                    st.info(f"📊 تعذّر الرسم: {e}")

        except Exception as e:
            st.error(f"❌ خطأ: {e}")
            st.info("💡 تأكد من صياغة المعادلة — راجع الأمثلة في الشريط الجانبي")


# ══════════════════════════════════════════════════════════
#  Footer
# ══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:rgba(255,255,255,0.2); font-size:12px; padding:10px">
    حل المعادلات التفاضلية العادية &nbsp;·&nbsp; Python + SymPy + Streamlit
</div>
""", unsafe_allow_html=True)
