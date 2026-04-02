# ═══════════════════════════════════════════════════════
#  solver.py  —  دوال الحل الرياضي
# ═══════════════════════════════════════════════════════
from sympy import *

x = symbols('x')
y = Function('y')

def get_order_and_type(ode_eq):
    """يحدد رتبة المعادلة وتصنيفها"""
    order = ode_order(ode_eq, y(x))
    cls   = classify_ode(ode_eq, y(x))
    return order, list(cls[:6])

def auto_solve_ode(ode_eq, ics=None):
    """يحل أي معادلة تفاضلية تلقائياً"""
    try:
        sol = dsolve(ode_eq, y(x), ics=ics) if ics else dsolve(ode_eq, y(x))
        return sol, None
    except Exception as e:
        return None, str(e)

def solve_separation(f_x_str, g_y_str):
    """فصل المتغيرات: dy/dx = f(x)*g(y)"""
    steps = []
    try:
        y_s = symbols('y')
        C1  = symbols('C1')
        f_x = sympify(f_x_str)
        g_y = sympify(g_y_str)

        steps.append(("التعرف على الشكل",
                       f"dy/dx = f(x)·g(y)",
                       f"f(x) = {f_x},   g(y) = {g_y}"))

        steps.append(("فصل المتغيرات",
                       f"dy / ({g_y}) = ({f_x}) dx",
                       "نقسم على g(y) وننقل dx للطرف الأيمن"))

        il = integrate(1/g_y, y_s)
        ir = integrate(f_x, x)
        steps.append(("التكامل من الطرفين",
                       f"∫ dy/({g_y}) = ∫ ({f_x}) dx",
                       f"{il} = {ir} + C"))

        implicit = Eq(il, ir + C1)
        steps.append(("الحل العام (ضمني)", str(implicit), ""))

        try:
            explicit = solve(implicit, y_s)
            steps.append(("الحل الصريح", f"y = {explicit}", ""))
        except:
            steps.append(("الحل الصريح", "لا يمكن إيجاد y صريحاً", ""))

        return steps, str(implicit)
    except Exception as e:
        return [("خطأ", str(e), "")], ""

def solve_linear_first(P_str, Q_str):
    """الخطية: dy/dx + P(x)y = Q(x)"""
    steps = []
    try:
        C1 = symbols('C1')
        P  = sympify(P_str)
        Q  = sympify(Q_str)

        steps.append(("الشكل القياسي",
                       "dy/dx + P(x)·y = Q(x)",
                       f"P(x) = {P},   Q(x) = {Q}"))

        mu_exp = integrate(P, x)
        mu     = simplify(exp(mu_exp))
        steps.append(("عامل التكامل",
                       f"μ(x) = e^(∫P dx) = e^({mu_exp})",
                       f"μ(x) = {mu}"))

        steps.append(("الضرب في μ",
                       f"d/dx[μ·y] = μ·Q = {simplify(mu*Q)}",
                       "الطرف الأيسر أصبح مشتقاً كاملاً"))

        rhs = integrate(simplify(mu * Q), x)
        sol = simplify((rhs + C1) / mu)
        steps.append(("التكامل",
                       f"μ·y = {rhs} + C",
                       ""))
        steps.append(("الحل العام",
                       f"y(x) = {sol}",
                       ""))

        return steps, str(sol)
    except Exception as e:
        return [("خطأ", str(e), "")], ""

def solve_bernoulli_method(P_str, Q_str, n_val):
    """برنولي: dy/dx + P(x)y = Q(x)y^n"""
    steps = []
    try:
        C1 = symbols('C1')
        P  = sympify(P_str)
        Q  = sympify(Q_str)
        n  = sympify(n_val)

        steps.append(("التعرف",
                       f"dy/dx + ({P})y = ({Q})·y^{n}",
                       f"n = {n}"))

        alpha = 1 - n
        steps.append(("التعويض v = y^(1-n)",
                       f"v = y^({alpha})",
                       f"dv/dx = {alpha}·y^(-{n})·dy/dx"))

        new_P = simplify(alpha * P)
        new_Q = simplify(alpha * Q)
        steps.append(("المعادلة الخطية الجديدة",
                       f"dv/dx + ({new_P})·v = {new_Q}",
                       "نطبق طريقة المعادلة الخطية"))

        mu_exp = integrate(new_P, x)
        mu     = simplify(exp(mu_exp))
        rhs    = integrate(simplify(mu * new_Q), x)
        v_sol  = simplify((rhs + C1) / mu)
        steps.append(("حل v",
                       f"v(x) = {v_sol}",
                       ""))

        y_sol = simplify(v_sol ** (1/alpha))
        steps.append(("الرجوع: y = v^(1/(1-n))",
                       f"y(x) = {y_sol}",
                       "⚠️ y = 0 حل ثابت أيضاً!"))

        return steps, str(y_sol)
    except Exception as e:
        return [("خطأ", str(e), "")], ""

def solve_exact_method(M_str, N_str):
    """التامة: M dx + N dy = 0"""
    steps = []
    try:
        y_s = symbols('y')
        C1  = symbols('C1')
        M   = sympify(M_str)
        N   = sympify(N_str)

        dM_dy = diff(M, y_s)
        dN_dx = diff(N, x)
        steps.append(("التحقق من التمامة",
                       f"∂M/∂y = {dM_dy}   |   ∂N/∂x = {dN_dx}",
                       "✅ تامة" if simplify(dM_dy - dN_dx) == 0 else "❌ غير تامة!"))

        if simplify(dM_dy - dN_dx) != 0:
            return steps, ""

        F       = integrate(M, x)
        dF_dy   = diff(F, y_s)
        g_prime = simplify(N - dF_dy)
        g_y     = integrate(g_prime, y_s)
        F_full  = simplify(F + g_y)

        steps.append(("تكامل M بالنسبة لـ x",
                       f"F = {F} + g(y)", ""))
        steps.append(("إيجاد g(y)",
                       f"g'(y) = {g_prime}   →   g(y) = {g_y}", ""))
        steps.append(("الحل: F(x,y) = C",
                       f"{F_full} = C", ""))

        return steps, f"{F_full} = C"
    except Exception as e:
        return [("خطأ", str(e), "")], ""

def solve_const_coeff(a_val, b_val, c_val):
    """معاملات ثابتة: ay'' + by' + cy = 0"""
    steps = []
    try:
        r       = symbols('r')
        C1, C2  = symbols('C1 C2')
        a, b, c = sympify(a_val), sympify(b_val), sympify(c_val)

        char_eq = a*r**2 + b*r + c
        roots   = solve(char_eq, r)
        delta   = b**2 - 4*a*c

        steps.append(("المعادلة المميزة",
                       f"{a}r² + {b}r + {c} = 0",
                       f"Δ = {delta}"))
        steps.append(("الجذور", f"r = {roots}", ""))

        if delta > 0:
            r1, r2 = roots
            sol  = C1*exp(r1*x) + C2*exp(r2*x)
            case = f"جذران حقيقيان مختلفان: r₁={r1}, r₂={r2}"
        elif delta == 0:
            r1  = roots[0]
            sol = (C1 + C2*x)*exp(r1*x)
            case = f"جذر مكرر: r = {r1}"
        else:
            r1         = roots[0]
            alpha_val  = re(r1)
            beta_val   = im(r1)
            sol = exp(alpha_val*x)*(C1*cos(beta_val*x)+C2*sin(beta_val*x))
            case = f"جذران مركبان: {alpha_val} ± {beta_val}i"

        steps.append(("الحالة", case, ""))
        steps.append(("الحل العام", f"y(x) = {sol}", ""))

        return steps, str(sol)
    except Exception as e:
        return [("خطأ", str(e), "")], ""

def solve_cauchy_euler_method(a_val, b_val, c_val):
    """كوشي-أويلر: ax²y'' + bxy' + cy = 0"""
    steps = []
    try:
        m       = symbols('m')
        C1, C2  = symbols('C1 C2')
        a, b, c = sympify(a_val), sympify(b_val), sympify(c_val)

        indicial = a*m*(m-1) + b*m + c
        roots    = solve(indicial, m)

        steps.append(("التعويض y = x^m",
                       f"{a}m(m-1) + {b}m + {c} = 0",
                       "المعادلة الإندائية"))
        steps.append(("الجذور", f"m = {roots}", ""))

        if len(roots) == 2 and roots[0] != roots[1]:
            m1, m2 = roots
            if im(m1) == 0:
                sol  = C1*x**m1 + C2*x**m2
                case = f"جذران حقيقيان: m₁={m1}, m₂={m2}"
            else:
                av   = re(m1); bv = im(m1)
                sol  = x**av*(C1*cos(bv*ln(x))+C2*sin(bv*ln(x)))
                case = f"جذران مركبان: {av} ± {bv}i"
        else:
            m1   = roots[0]
            sol  = x**m1*(C1 + C2*ln(x))
            case = f"جذر مكرر: m = {m1}"

        steps.append(("الحالة", case, ""))
        steps.append(("الحل العام", f"y(x) = {sol}", ""))

        return steps, str(sol)
    except Exception as e:
        return [("خطأ", str(e), "")], ""
