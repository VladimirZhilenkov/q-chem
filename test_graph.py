"""
test_graph.py
=============
Проверка нового LangGraph-агента на трёх уровнях:

  1. Структура графа    — узлы, рёбра, компиляция (без LLM, без PySCF)
  2. Отдельные узлы     — plan, generate_input, validate_and_convert (нужен LLM)
  3. Полный прогон      — весь пайплайн от запроса до отчёта (нужен LLM + PySCF)

Запуск:
    python test_graph.py                  # всё
    python test_graph.py --level 1        # только структура (быстро, без LLM)
    python test_graph.py --level 2        # структура + узлы (нужен LLM)
    python test_graph.py --level 3        # полный прогон (нужен LLM + PySCF)
    python test_graph.py --stream         # полный прогон с потоковым выводом
"""

import argparse
import json
import sys
import os

from dotenv import load_dotenv

load_dotenv()


# ── Helpers ──────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(msg: str):
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str):
    print(f"  {RED}✗{RESET} {msg}")


def info(msg: str):
    print(f"  {CYAN}ℹ{RESET} {msg}")


def header(title: str):
    print(f"\n{BOLD}{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}{RESET}\n")


# ── Level 1: Graph structure ────────────────────────────────────────────────

def test_structure():
    """Check that the graph compiles and has the expected topology."""
    header("Level 1 — Graph structure (no LLM needed)")

    try:
        from agent.graph import build_graph, AgentState
        ok("Imports work")
    except ImportError as e:
        fail(f"Import error: {e}")
        info("Make sure langgraph is installed: pip install langgraph")
        return False

    try:
        graph = build_graph()
        ok("Graph compiles without errors")
    except Exception as e:
        fail(f"Graph compilation failed: {e}")
        return False

    # Check node names
    expected_nodes = {
        "plan", "generate_input", "validate_and_convert",
        "execute", "error_recovery", "collect_result", "report",
    }
    actual_nodes = set(graph.get_graph().nodes.keys()) - {"__start__", "__end__"}

    missing = expected_nodes - actual_nodes
    extra = actual_nodes - expected_nodes

    if not missing:
        ok(f"All {len(expected_nodes)} expected nodes present")
    else:
        fail(f"Missing nodes: {missing}")

    if extra:
        info(f"Extra nodes (not a problem): {extra}")

    # Check that the graph has edges (basic sanity)
    edges = graph.get_graph().edges
    if len(edges) >= 7:
        ok(f"Graph has {len(edges)} edges (≥7 expected)")
    else:
        fail(f"Graph has only {len(edges)} edges — expected ≥7")

    # Print ASCII representation
    try:
        ascii_graph = graph.get_graph().draw_ascii()
        print(f"\n{ascii_graph}\n")
    except Exception:
        info("Could not render ASCII graph (optional dependency)")

    # Check AgentState has required fields
    required_fields = [
        "query", "calc_plan", "current_calc_index",
        "orca_input", "pyscf_script", "raw_output",
        "converged", "error_message", "results",
        "retry_count", "final_report",
    ]
    state_keys = set(AgentState.__annotations__.keys())
    missing_fields = [f for f in required_fields if f not in state_keys]
    if not missing_fields:
        ok(f"AgentState has all {len(required_fields)} required fields")
    else:
        fail(f"AgentState missing fields: {missing_fields}")

    return True


# ── Level 2: Individual nodes ───────────────────────────────────────────────

def test_nodes():
    """Test individual nodes with a simple water molecule query."""
    header("Level 2 — Individual nodes (needs LLM)")

    from agent.graph import (
        plan, generate_input, validate_and_convert,
        collect_result, report, AgentState,
    )

    query = "Calculate the B3LYP/def2-SVP energy of a water molecule (H2O)"

    # ── Test: plan ──
    print(f"  {YELLOW}Testing plan node...{RESET}")
    try:
        state = {"query": query}
        result = plan(state)

        assert "calc_plan" in result, "plan must return calc_plan"
        assert isinstance(result["calc_plan"], list), "calc_plan must be a list"
        assert len(result["calc_plan"]) >= 1, "calc_plan must have ≥1 entry"

        calc = result["calc_plan"][0]
        assert "method" in calc, "calc must have 'method'"
        assert "basis" in calc, "calc must have 'basis'"
        assert "atoms_xyz" in calc, "calc must have 'atoms_xyz'"

        ok(f"plan returned {len(result['calc_plan'])} calculation(s)")
        info(f"  method={calc['method']}, basis={calc['basis']}, "
             f"job_type={calc.get('job_type', '?')}")
        plan_result = result
    except Exception as e:
        fail(f"plan failed: {e}")
        return False

    # ── Test: generate_input ──
    print(f"\n  {YELLOW}Testing generate_input node...{RESET}")
    try:
        state = {
            **plan_result,
            "query": query,
            "error_message": "",
            "orca_input": "",
            "results": [],
        }
        result = generate_input(state)

        assert "orca_input" in result, "must return orca_input"
        orca = result["orca_input"]
        assert len(orca) > 20, "orca_input seems too short"
        assert "!" in orca, "ORCA input must have ! keyword line"
        assert "*" in orca, "ORCA input must have * geometry block"

        ok(f"generate_input produced {len(orca)} chars of ORCA input")

        # Show first few lines
        lines = orca.strip().split("\n")
        for line in lines[:5]:
            info(f"  {line}")
        if len(lines) > 5:
            info(f"  ... ({len(lines) - 5} more lines)")

        gen_result = result
    except Exception as e:
        fail(f"generate_input failed: {e}")
        return False

    # ── Test: validate_and_convert ──
    print(f"\n  {YELLOW}Testing validate_and_convert node...{RESET}")
    try:
        state = {"orca_input": gen_result["orca_input"]}
        result = validate_and_convert(state)

        if result.get("error_message"):
            fail(f"Conversion error: {result['error_message']}")
            return False

        pyscf = result["pyscf_script"]
        assert "from pyscf" in pyscf, "PySCF script must import pyscf"
        assert "mol.atom" in pyscf or "mol.build" in pyscf, "must define molecule"

        ok(f"validate_and_convert produced {len(pyscf)} chars of PySCF script")

        lines = pyscf.strip().split("\n")
        for line in lines[:5]:
            info(f"  {line}")
        if len(lines) > 5:
            info(f"  ... ({len(lines) - 5} more lines)")

    except Exception as e:
        fail(f"validate_and_convert failed: {e}")
        return False

    # ── Test: collect_result (with mock data) ──
    print(f"\n  {YELLOW}Testing collect_result node (mock data)...{RESET}")
    try:
        state = {
            "calc_plan": plan_result["calc_plan"],
            "current_calc_index": 0,
            "raw_output": "converged SCF energy = -76.0234567890 Hartree\nconverged = True",
            "converged": True,
            "error_message": "",
            "results": [],
        }
        result = collect_result(state)

        assert "results" in result, "must return results"
        assert len(result["results"]) == 1, "must have 1 result"
        assert result["current_calc_index"] == 1, "must advance index"
        assert result["results"][0]["energy_hartree"] is not None, "must parse energy"

        ok(f"collect_result parsed energy: {result['results'][0]['energy_hartree']} Ha")
    except Exception as e:
        fail(f"collect_result failed: {e}")
        return False

    # ── Test: report (with mock data) ──
    print(f"\n  {YELLOW}Testing report node (mock results)...{RESET}")
    try:
        state = {
            "query": query,
            "results": [{
                "step": 0,
                "description": "B3LYP/def2-SVP energy of water",
                "method": "B3LYP",
                "basis": "def2-SVP",
                "job_type": "energy",
                "energy_hartree": -76.0234567890,
                "converged": True,
            }],
        }
        result = report(state)

        assert "final_report" in result, "must return final_report"
        assert len(result["final_report"]) > 50, "report seems too short"

        ok(f"report generated ({len(result['final_report'])} chars)")
        info(f"  First 200 chars: {result['final_report'][:200]}...")
    except Exception as e:
        fail(f"report failed: {e}")
        return False

    return True


# ── Level 3: Full pipeline ──────────────────────────────────────────────────

def test_full(stream: bool = False):
    """Run the complete agent on a simple query."""
    header("Level 3 — Full pipeline (needs LLM + PySCF)")

    try:
        import pyscf  # noqa: F401
        ok("PySCF is importable")
    except ImportError:
        fail("PySCF is not installed — pip install pyscf")
        info("Skipping full pipeline test")
        return False

    query = "Calculate the HF/STO-3G energy of H2 molecule (bond length 0.74 Å)"

    if stream:
        print(f"\n  {YELLOW}Streaming agent...{RESET}\n")
        from agent.graph import stream_agent

        final_report = None
        for event in stream_agent(query):
            for node_name, state_update in event.items():
                print(f"  {CYAN}[{node_name}]{RESET}")

                if node_name == "plan":
                    plan = state_update.get("calc_plan", [])
                    info(f"  Plan: {len(plan)} calculation(s)")
                    for i, c in enumerate(plan):
                        info(f"    {i}: {c.get('method')}/{c.get('basis')} "
                             f"({c.get('job_type')})")

                elif node_name == "generate_input":
                    orca = state_update.get("orca_input", "")
                    info(f"  ORCA input: {len(orca)} chars")

                elif node_name == "validate_and_convert":
                    err = state_update.get("error_message", "")
                    if err:
                        fail(f"  Conversion error: {err}")
                    else:
                        pyscf_len = len(state_update.get("pyscf_script", ""))
                        info(f"  PySCF script: {pyscf_len} chars")

                elif node_name == "execute":
                    conv = state_update.get("converged", False)
                    err = state_update.get("error_message", "")
                    if conv:
                        ok("  Calculation converged")
                    elif err:
                        fail(f"  {err[:120]}")

                elif node_name == "error_recovery":
                    retry = state_update.get("retry_count", 0)
                    info(f"  Retry {retry}/{3}")

                elif node_name == "collect_result":
                    results = state_update.get("results", [])
                    if results:
                        last = results[-1]
                        e = last.get("energy_hartree")
                        info(f"  Energy: {e} Ha" if e else "  Energy: not parsed")

                elif node_name == "report":
                    final_report = state_update.get("final_report", "")
                    ok(f"  Report generated ({len(final_report)} chars)")

        if final_report:
            print(f"\n{'─' * 60}")
            print(final_report)
            print(f"{'─' * 60}")

    else:
        print(f"\n  {YELLOW}Running agent (invoke)...{RESET}\n")
        from agent.graph import run_agent

        try:
            result = run_agent(query)
            ok(f"Agent completed — report length: {len(result)} chars")
            print(f"\n{'─' * 60}")
            print(result)
            print(f"{'─' * 60}")
        except Exception as e:
            fail(f"Agent failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test the LangGraph quantum chemistry agent")
    parser.add_argument(
        "--level", type=int, choices=[1, 2, 3], default=None,
        help="Test level: 1=structure, 2=nodes, 3=full pipeline (default: all)",
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="Use streaming mode for level 3",
    )
    args = parser.parse_args()

    results = {}

    if args.level is None or args.level == 1:
        results["structure"] = test_structure()

    if args.level is None or args.level == 2:
        results["nodes"] = test_nodes()

    if args.level is None or args.level == 3:
        results["full"] = test_full(stream=args.stream)

    # Summary
    header("Summary")
    all_passed = True
    for name, passed in results.items():
        if passed:
            ok(f"{name}")
        else:
            fail(f"{name}")
            all_passed = False

    if all_passed:
        print(f"\n  {GREEN}{BOLD}All tests passed!{RESET}\n")
    else:
        print(f"\n  {RED}{BOLD}Some tests failed.{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()