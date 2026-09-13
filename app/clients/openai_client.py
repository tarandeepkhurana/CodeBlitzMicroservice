"""
OpenAI Client for GPT-4o variant generation.

Uses JSON mode for structured output generation.
"""

import copy
import json
import logging
from typing import Optional

from openai import AsyncOpenAI

from app.config import settings
from app.core import get_metrics

logger = logging.getLogger(__name__)
metrics = get_metrics()


# ═══════════════════════════════════════════════════════════════════════════════
# System Prompts - COMPREHENSIVE WITH REAL WORKING EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════

VARIANT_GENERATION_SYSTEM_PROMPT = """You generate coding question VARIANTS for a competitive programming platform.
You receive a base question with WORKING code - you must create a variant that also WORKS.

════════════════════════════════════════════════════════════════════════════════
⚠️ CRITICAL: JAVA SCANNER CAUSES TIMEOUTS! ⚠️
════════════════════════════════════════════════════════════════════════════════
NEVER use java.util.Scanner! Scanner.nextInt() BLOCKS FOREVER waiting for input!
ALWAYS use BufferedReader:
  BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
  int n = Integer.parseInt(br.readLine().trim());

════════════════════════════════════════════════════════════════════════════════
🚨 ABSOLUTE RULES - VIOLATING THESE CAUSES FAILURES 🚨
════════════════════════════════════════════════════════════════════════════════

1. STDIN FORMAT: Use EXACT same format as base_question.test_cases_sample
   - If base uses "1 8 6 2 5 4 8 3 7" → you use single line space-separated
   - If base uses "1 2 4\\n1 3 4" → you use two lines
   - DO NOT add size prefixes or change format!

2. COPY BASE WRAPPERS: Your stdin_wrappers should be nearly identical to base
   - Only change function/method names
   - DO NOT rewrite parsing logic unless absolutely necessary

3. 🔴🔴🔴 JAVA CRITICAL - READ THIS CAREFULLY 🔴🔴🔴
   - {user_solution} MUST be AFTER the closing brace of Main class, NOT inside it!
   - If solution is inside Main, it becomes an INNER CLASS and static methods FAIL!
   
   ❌ WRONG (causes "Illegal static declaration in inner class"):
   public class Main {
       {user_solution}  // WRONG! Inside Main = inner class
       public static void main(...) { ... }
   }
   
   ✅ CORRECT:
   public class Main {
       public static void main(...) {
           Solution sol = new Solution();
           System.out.println(sol.methodName(args));
           System.exit(0);  // CRITICAL: Force JVM exit!
       }
   }
   
   {user_solution}  // CORRECT! Outside Main = separate class
   
   - NEVER use Scanner (Scanner causes timeouts - use BufferedReader!)
   - Read ONLY the lines you need, then print and CALL System.exit(0)!

4. PLACEHOLDER: Use {user_solution} exactly (lowercase, underscore)

5. TEST CASES: Exactly 10 (3 visible is_hidden:false, 7 hidden is_hidden:true)
   - Use SMALL, simple inputs (arrays ≤10 elements, integers ≤1000, strings ≤20 chars)
   - Focus on EDGE CASES: empty input, single element, min/max bounds, duplicates
   - NO stress tests or large inputs - test correctness, not performance
   - Keep expected_stdout simple (single number, "true"/"false", short string)

6. ALL 3 LANGUAGES: python, java, cpp - MUST have all three

7. CROSS-CHECK ORACLE: you do NOT write one. A separate request will write a
   brute-force oracle from your problem_statement ALONE - it never sees your
   solution_code - and the two must agree on every test and on hundreds of
   random inputs.
   => Your problem_statement must define the answer EXACTLY: every edge case,
      sign convention, tie-break and output format. If a careful reader could
      implement something different from your solution_code, the question is
      rejected. Never hide a rule (like "return the absolute value") in the
      solution only.

8. GENERATOR: also return "test_generator_code": {"python": "..."} defining
     import random
     def generate_test_case(size='small', edge_case=None):
         ...
         return stdin_string
   - Returns ONE random VALID input as a stdin string, in exactly the test_cases
     stdin format. Use the `random` module (it is seeded for you).
   - size='small' means tiny inputs (e.g. n <= 8, values within about -10..10)
     so the brute force finishes instantly. Respect EVERY constraint
     (sortedness, uniqueness, ranges, guaranteed-answer promises...).

9. IT MUST BE A DIFFERENT PROBLEM, NOT A RENAME
   - The base question's OWN solution is run against YOUR test cases. If it still
     passes them all, your variant is thrown away.
   - Renaming the function, relabelling values (e.g. digits 1-9 -> 0-8), swapping
     the story or rewording the statement does NOT count.
   - Change a RULE instead: what must be returned, which inputs are valid, an
     extra condition to satisfy, a tie-break, or tighter limits that force a
     better algorithm.

10. expected_stdout in test_cases is only YOUR PREDICTION. The real expected
   outputs are computed by EXECUTING your solution_code, which is then
   cross-checked against brute_force_solution on hundreds of random inputs
   and against the Java/C++ solutions. A wrong solution gets the question
   rejected - make solution_code exactly match the problem statement.

════════════════════════════════════════════════════════════════════════════════
WORKING EXAMPLE 1: Single Array Input → Integer Output
════════════════════════════════════════════════════════════════════════════════
Problem: Container With Most Water
stdin format: "1 8 6 2 5 4 8 3 7" (space-separated integers on ONE line)
expected_stdout: "49"

test_cases examples:
  {"stdin": "1 8 6 2 5 4 8 3 7", "expected_stdout": "49", "is_hidden": false}
  {"stdin": "1 1", "expected_stdout": "1", "is_hidden": false}
  {"stdin": "5 5 5 5 5", "expected_stdout": "20", "is_hidden": true}

solution_code:
  python: "def maxArea(height):\\n    left, right = 0, len(height) - 1\\n    max_area = 0\\n    while left < right:\\n        width = right - left\\n        h = min(height[left], height[right])\\n        max_area = max(max_area, width * h)\\n        if height[left] < height[right]:\\n            left += 1\\n        else:\\n            right -= 1\\n    return max_area"
  java: "class Solution {\\n    int maxArea(int[] height) {\\n        int left = 0, right = height.length - 1;\\n        int maxArea = 0;\\n        while (left < right) {\\n            int width = right - left;\\n            int h = Math.min(height[left], height[right]);\\n            maxArea = Math.max(maxArea, width * h);\\n            if (height[left] < height[right]) left++;\\n            else right--;\\n        }\\n        return maxArea;\\n    }\\n}"
  cpp: "int maxArea(vector<int>& height) {\\n    int left = 0, right = height.size() - 1;\\n    int maxArea = 0;\\n    while (left < right) {\\n        int width = right - left;\\n        int h = min(height[left], height[right]);\\n        maxArea = max(maxArea, width * h);\\n        if (height[left] < height[right]) left++;\\n        else right--;\\n    }\\n    return maxArea;\\n}"

stdin_wrappers:
  python: "import sys\\n\\n{user_solution}\\n\\nif __name__ == '__main__':\\n    data = sys.stdin.read().strip()\\n    height = list(map(int, data.split()))\\n    result = maxArea(height)\\n    print(result)"
  java: "import java.io.*;\\nimport java.util.*;\\n\\npublic class Main {\\n    public static void main(String[] args) throws IOException {\\n        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));\\n        String line = br.readLine();\\n        String[] parts = line.trim().split(\\"\\\\\\\\s+\\");\\n        int[] height = new int[parts.length];\\n        for (int i = 0; i < parts.length; i++) height[i] = Integer.parseInt(parts[i]);\\n        Solution sol = new Solution();\\n        System.out.println(sol.maxArea(height));\\n        System.exit(0);\\n    }\\n}\\n\\n{user_solution}"
  cpp: "#include <iostream>\\n#include <vector>\\n#include <algorithm>\\nusing namespace std;\\n\\n{user_solution}\\n\\nint main() {\\n    vector<int> height;\\n    int x;\\n    while (cin >> x) height.push_back(x);\\n    cout << maxArea(height) << endl;\\n    return 0;\\n}"

function_template:
  python: "def maxArea(height):\\n    # Your code here\\n    pass"
  java: "class Solution {\\n    int maxArea(int[] height) {\\n        // Your code here\\n        return 0;\\n    }\\n}"
  cpp: "int maxArea(vector<int>& height) {\\n    // Your code here\\n    return 0;\\n}"

════════════════════════════════════════════════════════════════════════════════
WORKING EXAMPLE 2: Single Integer Input → Boolean Output
════════════════════════════════════════════════════════════════════════════════
Problem: Palindrome Number
stdin format: "121" (single integer)
expected_stdout: "true" or "false" (lowercase!)

test_cases examples:
  {"stdin": "121", "expected_stdout": "true", "is_hidden": false}
  {"stdin": "-121", "expected_stdout": "false", "is_hidden": false}
  {"stdin": "10", "expected_stdout": "false", "is_hidden": true}

solution_code:
  python: "def isPalindrome(x):\\n    if x < 0:\\n        return False\\n    if x != 0 and x % 10 == 0:\\n        return False\\n    reversed_half = 0\\n    while x > reversed_half:\\n        reversed_half = reversed_half * 10 + x % 10\\n        x //= 10\\n    return x == reversed_half or x == reversed_half // 10"
  java: "class Solution {\\n    public boolean isPalindrome(int x) {\\n        if (x < 0 || (x != 0 && x % 10 == 0)) return false;\\n        int reversedHalf = 0;\\n        while (x > reversedHalf) {\\n            reversedHalf = reversedHalf * 10 + x % 10;\\n            x /= 10;\\n        }\\n        return x == reversedHalf || x == reversedHalf / 10;\\n    }\\n}"
  cpp: "bool isPalindrome(int x) {\\n    if (x < 0 || (x != 0 && x % 10 == 0)) return false;\\n    int reversed_half = 0;\\n    while (x > reversed_half) {\\n        reversed_half = reversed_half * 10 + x % 10;\\n        x /= 10;\\n    }\\n    return x == reversed_half || x == reversed_half / 10;\\n}"

stdin_wrappers:
  python: "import sys\\n{user_solution}\\nx = int(input())\\nresult = isPalindrome(x)\\nprint('true' if result else 'false')"
  java: "import java.io.*;\\npublic class Main {\\n    public static void main(String[] args) throws IOException {\\n        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));\\n        int x = Integer.parseInt(br.readLine().trim());\\n        Solution sol = new Solution();\\n        System.out.println(sol.isPalindrome(x) ? \\"true\\" : \\"false\\");\\n        System.exit(0);\\n    }\\n}\\n{user_solution}"
  cpp: "#include <iostream>\\nusing namespace std;\\n{user_solution}\\nint main() {\\n    int x;\\n    cin >> x;\\n    cout << (isPalindrome(x) ? \\"true\\" : \\"false\\") << endl;\\n    return 0;\\n}"

════════════════════════════════════════════════════════════════════════════════
WORKING EXAMPLE 3: Two Arrays Input (Two Lines) → Array Output
════════════════════════════════════════════════════════════════════════════════
Problem: Merge Two Sorted Lists
stdin format: "1 2 4\\n1 3 4" (line1: first array, line2: second array)
expected_stdout: "1 1 2 3 4 4" (space-separated, empty string if empty result)

test_cases examples:
  {"stdin": "1 2 4\\n1 3 4", "expected_stdout": "1 1 2 3 4 4", "is_hidden": false}
  {"stdin": "\\n", "expected_stdout": "", "is_hidden": false}
  {"stdin": "1 2 3\\n4 5 6", "expected_stdout": "1 2 3 4 5 6", "is_hidden": true}

stdin_wrappers (with helper classes for linked lists):
  python: "import sys\\n\\nclass ListNode:\\n    def __init__(self, val=0, next=None):\\n        self.val = val\\n        self.next = next\\n\\ndef list_to_linked(arr):\\n    if not arr:\\n        return None\\n    head = ListNode(arr[0])\\n    curr = head\\n    for val in arr[1:]:\\n        curr.next = ListNode(val)\\n        curr = curr.next\\n    return head\\n\\ndef linked_to_list(head):\\n    result = []\\n    while head:\\n        result.append(head.val)\\n        head = head.next\\n    return result\\n\\n{user_solution}\\n\\nif __name__ == '__main__':\\n    lines = sys.stdin.read().split('\\\\n')\\n    line1 = lines[0].strip() if len(lines) > 0 else ''\\n    line2 = lines[1].strip() if len(lines) > 1 else ''\\n    arr1 = list(map(int, line1.split())) if line1 else []\\n    arr2 = list(map(int, line2.split())) if line2 else []\\n    list1 = list_to_linked(arr1)\\n    list2 = list_to_linked(arr2)\\n    merged = mergeTwoLists(list1, list2)\\n    result = linked_to_list(merged)\\n    print(' '.join(map(str, result)) if result else '')"
  java: "import java.io.*;\\nimport java.util.*;\\n\\npublic class Main {\\n    public static void main(String[] args) throws IOException {\\n        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));\\n        String line1 = br.readLine();\\n        String line2 = br.readLine();\\n        ListNode list1 = arrayToList(parseLine(line1));\\n        ListNode list2 = arrayToList(parseLine(line2));\\n        Solution sol = new Solution();\\n        ListNode merged = sol.mergeTwoLists(list1, list2);\\n        StringBuilder sb = new StringBuilder();\\n        while (merged != null) {\\n            if (sb.length() > 0) sb.append(\\" \\");\\n            sb.append(merged.val);\\n            merged = merged.next;\\n        }\\n        System.out.println(sb.toString());\\n    }\\n    static int[] parseLine(String line) {\\n        if (line == null || line.trim().isEmpty()) return new int[0];\\n        String[] parts = line.trim().split(\\"\\\\\\\\s+\\");\\n        int[] arr = new int[parts.length];\\n        for (int i = 0; i < parts.length; i++) arr[i] = Integer.parseInt(parts[i]);\\n        return arr;\\n    }\\n    static ListNode arrayToList(int[] arr) {\\n        if (arr.length == 0) return null;\\n        ListNode head = new ListNode(arr[0]);\\n        ListNode curr = head;\\n        for (int i = 1; i < arr.length; i++) {\\n            curr.next = new ListNode(arr[i]);\\n            curr = curr.next;\\n        }\\n        return head;\\n    }\\n}\\nclass ListNode {\\n    int val;\\n    ListNode next;\\n    ListNode() {}\\n    ListNode(int val) { this.val = val; }\\n}\\n{user_solution}"

════════════════════════════════════════════════════════════════════════════════
OUTPUT JSON SCHEMA (REQUIRED STRUCTURE)
════════════════════════════════════════════════════════════════════════════════
{
  "title": "Variant Title (describe the modification)",
  "problem_statement": "Full problem description with examples",
  "modifications": ["what you changed from base question"],
  "input_format": {"params": [{"name": "x", "type": "int[]", "description": "..."}]},
  "output_format": {"type": "int", "description": "..."},
  "constraints": [{"text": "1 <= n <= 100", "variable": "n", "min": 1, "max": 100}],
  "examples": [{"input": {"param": [1,2,3]}, "output": 6, "explanation": "..."}],
  "solution_code": {
    "python": "def functionName(params):\\n    # implementation",
    "java": "class Solution {\\n    returnType methodName(params) {\\n        // implementation\\n    }\\n}",
    "cpp": "returnType functionName(params) {\\n    // implementation\\n}"
  },
  "stdin_wrappers": {
    "python": "import sys\\n{user_solution}\\n...",
    "java": "import java.io.*;\\npublic class Main {...}\\n{user_solution}",
    "cpp": "#include <iostream>\\n{user_solution}\\nint main() {...}"
  },
  "function_template": {
    "python": "def functionName(params):\\n    pass",
    "java": "class Solution {\\n    returnType methodName(params) {\\n        return ...;\\n    }\\n}",
    "cpp": "returnType functionName(params) {\\n    return ...;\\n}"
  },
  "test_generator_code": {
    "python": "import random\\n\\ndef generate_test_case(size='small', edge_case=None):\\n    ...\\n    return stdin_string"
  },
  "edge_case_types": ["empty_input", "single_element", "duplicates", "..."],
  "solution_explanation": "Algorithm description",
  "expected_time_complexity": "O(n)",    // ⚠️ MUST be from STANDARD LIST below!
  "expected_space_complexity": "O(1)",   // ⚠️ MUST be from STANDARD LIST below!
  "test_cases": [
    {"stdin": "...", "expected_stdout": "...", "is_hidden": false},
    {"stdin": "...", "expected_stdout": "...", "is_hidden": false},
    {"stdin": "...", "expected_stdout": "...", "is_hidden": false},
    {"stdin": "...", "expected_stdout": "...", "is_hidden": true},
    {"stdin": "...", "expected_stdout": "...", "is_hidden": true},
    {"stdin": "...", "expected_stdout": "...", "is_hidden": true},
    {"stdin": "...", "expected_stdout": "...", "is_hidden": true},
    {"stdin": "...", "expected_stdout": "...", "is_hidden": true},
    {"stdin": "...", "expected_stdout": "...", "is_hidden": true},
    {"stdin": "...", "expected_stdout": "...", "is_hidden": true}
  ]
}

════════════════════════════════════════════════════════════════════════════════
⚠️ COMPLEXITY VALUES - USE ONLY THESE EXACT STRINGS
════════════════════════════════════════════════════════════════════════════════
Pick expected_time_complexity and expected_space_complexity ONLY from this list:
  "O(1)", "O(1)*", "O(log n)", "O(log²n)", "O(log³n)", "O(√n)", "O(n)",
  "O(n log log n)", "O(n log n)", "O(n+m)", "O(min(n,m))", "O(n*m)",
  "O((V+E)log V)", "O(n log k)", "O(n²)", "O(n² log n)", "O(n³)",
  "O(V+E)", "O(2^n)", "O(n!)", "O(n!/(k!(n-k)!))", "O(recursive)"

DO NOT write verbose descriptions like "O(n) where n is array length"!
DO NOT invent new complexity notations!
Just pick the closest standard notation from the list above.

═══════════════════════════════════════════════════════════════════════════════

════════════════════════════════════════════════════════════════════════════════
VERIFICATION CHECKLIST (before returning JSON)
════════════════════════════════════════════════════════════════════════════════
□ stdin format matches base_question.test_cases_sample exactly
□ stdin_wrappers parse the stdin correctly for all 3 languages
□ Java uses BufferedReader (NOT Scanner)
□ All wrappers use {user_solution} placeholder
□ solution_code is correct for all 3 languages
□ problem_statement alone is enough to implement solution_code exactly (an
  independent solver reading only the statement must produce the same outputs)
□ test_generator_code returns valid small stdin strings in the exact test format
□ Ran solution mentally on test cases - expected_stdout is correct
□ Exactly 10 test_cases (3 visible, 7 hidden)
□ Output format matches (lowercase "true"/"false" for boolean, space-separated for arrays)
"""

# Fix prompt for when verification fails
FIX_VARIANT_SYSTEM_PROMPT = """You fix coding problem variants that failed PISTON execution verification.

════════════════════════════════════════════════════════════════════════════════
HOW VERIFICATION WORKS (read first)
════════════════════════════════════════════════════════════════════════════════
- Expected outputs are COMPUTED by running solution_code.python on each test
  input. You cannot fix anything by editing expected_stdout - it is ignored.
- brute_force_solution.python must print the same output as solution_code.python
  on every test and on random inputs from test_generator_code. When they
  disagree, re-read the problem statement and decide which one is wrong; fix
  that one (or make the statement unambiguous if both readings are plausible).
- solution_code.java and solution_code.cpp must print EXACTLY what Python
  prints (same format, same spacing, lowercase true/false).
- Failure kinds you may see: reference_error (Python solution crashed),
  brute_mismatch, brute_force_error, generator_error, language_mismatch,
  harness_incompatible (Python wrapper must read input via sys.stdin/input(),
  never open(0)), too_few_random_cases (make 'small' inputs tinier or the
  brute force faster), brute_force_too_slow / reference_too_slow (one input
  used too much CPU - the brute force is an ORACLE and may use any forbidden
  operator, e.g. // and %, instead of looping millions of times),
  template_mismatch (function_template must compile inside the wrapper -
  players start from it), missing_fields.

════════════════════════════════════════════════════════════════════════════════
⚠️ CRITICAL JAVA ISSUE: SCANNER CAUSES TIMEOUTS! ⚠️
════════════════════════════════════════════════════════════════════════════════

NEVER use java.util.Scanner in Java wrappers!
Scanner.nextInt(), Scanner.nextLine() etc. BLOCK FOREVER waiting for input even
after the program produces correct output!

❌ FORBIDDEN - WILL TIMEOUT:
Scanner sc = new Scanner(System.in);
int n = sc.nextInt();  // BLOCKS FOREVER!

✅ REQUIRED - USE BUFFEREDREADER:
BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
int n = Integer.parseInt(br.readLine().trim());  // Reads, finishes, exits!

If you see "Runtime timeout" but actual output is CORRECT, 99% chance it's Scanner!

════════════════════════════════════════════════════════════════════════════════
COMMON FAILURE PATTERNS AND FIXES
════════════════════════════════════════════════════════════════════════════════

1. "Illegal static declaration in inner class" (JAVA COMPILATION ERROR)
   → {user_solution} is INSIDE the Main class, making Solution an inner class
   → Inner classes CANNOT have static methods in Java!
   → FIX: Move {user_solution} AFTER the closing brace of Main class
   
   ❌ WRONG:
   public class Main {
       {user_solution}  // Makes Solution an INNER class!
       public static void main(...) {}
   }
   
   ✅ CORRECT:
   public class Main {
       public static void main(...) {
           Solution sol = new Solution();
           System.out.println(sol.method());
           System.exit(0);  // CRITICAL!
       }
   }
   
   {user_solution}  // OUTSIDE Main - separate class!

2. "Runtime timeout" with CORRECT actual output
   → SCANNER BLOCKING OR MISSING System.exit(0)!
   → FIX: Replace Scanner with BufferedReader AND add System.exit(0) after println!

3. "Runtime timeout" with EMPTY actual output
   → Infinite loop OR wrapper blocking before output
   → FIX: Check solution logic, check input parsing

4. actual != expected (no error)
   → Solution logic wrong OR expected_stdout wrong
   → FIX: Either fix solution_code OR fix test_cases

════════════════════════════════════════════════════════════════════════════════
JAVA WRAPPER TEMPLATE (CORRECT STRUCTURE - NO SCANNER!)
════════════════════════════════════════════════════════════════════════════════

import java.io.*;
import java.util.*;

public class Main {
    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String line = br.readLine();  // Read ONLY what you need
        // Parse input
        Solution sol = new Solution();
        System.out.println(sol.methodName(args));  // Print result
        System.exit(0);  // ⚠️ CRITICAL: Force JVM to exit immediately!
    }
}

{user_solution}   // <-- MUST BE HERE, AFTER Main class closes!

════════════════════════════════════════════════════════════════════════════════
RULES FOR FIXES
════════════════════════════════════════════════════════════════════════════════
- If fixing test_cases: MUST return EXACTLY 10 test cases (never fewer!)
- If fixing solution_code: Return ALL 3 languages (python, java, cpp)
- If fixing stdin_wrappers: Return ALL 3 languages
- Preserve the stdin FORMAT (don't change format, just fix parsing)
- Double-check your fix actually addresses the failure

🔴 CRITICAL: PRESERVE {user_solution} PLACEHOLDER 🔴
- When returning stdin_wrappers, you MUST include the literal text {user_solution}
- This is where user code gets inserted - WITHOUT IT THE WRAPPER IS BROKEN
- Copy the exact position of {user_solution} from the original wrapper
- For Java: {user_solution} MUST appear AFTER the closing brace of Main class

════════════════════════════════════════════════════════════════════════════════
RETURN FORMAT
════════════════════════════════════════════════════════════════════════════════
Return JSON with ONLY the fields that need fixing:

{"solution_code": {"python": "...", "java": "...", "cpp": "..."}}
{"stdin_wrappers": {"python": "...", "java": "...", "cpp": "..."}}
{"brute_force_solution": {"python": "..."}}
{"test_generator_code": {"python": "..."}}
{"function_template": {"python": "...", "java": "...", "cpp": "..."}}
{"test_cases": [at least 10 test cases - only if some INPUTS are invalid]}
{"problem_statement": "..."}  (only to remove an ambiguity)
OR any combination of the above

DO NOT return fields that are working correctly."""


def _python_field(value) -> Optional[str]:
    """Code from a {"python": code} field or a plain string."""
    if isinstance(value, dict):
        value = value.get("python")
    return value if isinstance(value, str) and value.strip() else None


def _describe_failures(failures: list[dict], max_items: int = 10) -> str:
    """Render verification failures (see app/services/verification.py) for the fix prompt."""
    lines = []
    for f in failures[:max_items]:
        kind = f.get("kind", "unknown")
        where = f"test #{f['index']}" if "index" in f else (
            f"test #{f['test']}" if "test" in f else (f"random input (seed {f['seed']})" if "seed" in f else ""))
        if kind == "brute_mismatch":
            lines.append(
                f"- brute_mismatch on {where}: solution_code.python and brute_force_solution disagree.\n"
                f"    Input: {f.get('stdin')!r}\n    solution_code.python printed: {f.get('reference')!r}\n"
                f"    brute_force_solution printed: {f.get('brute_force')!r}")
        elif kind == "language_mismatch":
            lines.append(
                f"- language_mismatch: {f.get('language', '').upper()} differs from Python on {where}.\n"
                f"    Input: {f.get('stdin')!r}\n    Python printed: {f.get('expected')!r}\n"
                f"    {f.get('language')} printed: {f.get('actual')!r}\n    Error: {f.get('error') or 'none'}")
        elif kind == "missing_fields":
            lines.append(f"- missing_fields: {f.get('detail')}")
        elif kind == "template_mismatch":
            lang = f.get("language")
            lines.append(
                f"- template_mismatch: function_template.{lang} does not compile inside stdin_wrappers.{lang} "
                f"(players start from this template). Make them agree: same function/method name, "
                f"static-ness, parameter and return types.\n    Error: {f.get('error')}")
        elif kind == "harness_incompatible":
            lines.append("- harness_incompatible: the Python wrapper must read input via sys.stdin / input() (not open(0)).")
        elif kind == "too_few_random_cases":
            lines.append(f"- too_few_random_cases: only {f.get('checked')} random inputs checked "
                         f"({f.get('stopped')}); make size='small' inputs tinier or the brute force faster.")
        else:
            lines.append(f"- {kind} {where}: input {f.get('stdin')!r} error: {f.get('error')}")
    if len(failures) > max_items:
        lines.append(f"- ... and {len(failures) - max_items} more failures")
    return "\n".join(lines)


class OpenAIClient:
    """Async OpenAI client - supports both regular OpenAI and Azure AI Foundry."""
    
    def __init__(self):
        self._client: Optional[AsyncOpenAI] = None
        self._use_azure = settings.USE_AZURE_AI_FOUNDRY
    
    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            # 120 second timeout to prevent hanging
            timeout = 120.0
            
            if self._use_azure and settings.AZURE_AI_FOUNDRY_API_KEY:
                # Use Azure AI Foundry (OpenAI-compatible endpoint)
                base_url = f"{settings.AZURE_AI_FOUNDRY_ENDPOINT.rstrip('/')}/openai/v1/"
                self._client = AsyncOpenAI(
                    base_url=base_url,
                    api_key=settings.AZURE_AI_FOUNDRY_API_KEY,
                    timeout=timeout
                )
                logger.info(f"Using Azure AI Foundry: {settings.AZURE_AI_FOUNDRY_ENDPOINT}")
            else:
                # Use regular OpenAI API
                self._client = AsyncOpenAI(
                    api_key=settings.OPENAI_API_KEY,
                    timeout=timeout
                )
                logger.info("Using regular OpenAI API")
        return self._client
    
    @property
    def client(self) -> AsyncOpenAI:
        return self._get_client()
    
    async def generate_variant(
        self,
        base_question: dict,
        user_context: dict,
        generation_params: dict = None
    ) -> Optional[dict]:
        """
        Generate a question variant using GPT-4o.
        
        Args:
            base_question: The original question data (full with stdin_wrappers, etc.)
            user_context: User skill information (avg_rating, weak_areas)
            generation_params: Optional params like difficulty_increase
        
        Returns:
            Parsed variant data or None on failure
        """
        generation_params = generation_params or {
            "target_difficulty_increase": 1,
            "focus_areas": ["edge_cases", "constraints"]
        }
        
        # Filter to only Python/Java/C++ (the 3 languages we support)
        langs = ["python", "java", "cpp"]
        
        def filter_langs(d):
            if isinstance(d, dict):
                return {k: v for k, v in d.items() if k in langs}
            return d
        
        # Send COMPLETE base question data - LLM needs to see what works
        user_prompt = json.dumps({
            "base_question": {
                "title": base_question.get("title"),
                "problem_statement": base_question.get("problem_statement"),
                "difficulty": base_question.get("difficulty"),
                "category": base_question.get("category"),
                "input_format": base_question.get("input_format"),
                "output_format": base_question.get("output_format"),
                "constraints": base_question.get("constraints"),
                "examples": base_question.get("examples"),
                "expected_time_complexity": base_question.get("expected_time_complexity"),
                "expected_space_complexity": base_question.get("expected_space_complexity"),
                # FULL working code that passes PISTON - LLM should follow same patterns
                "solution_code": filter_langs(base_question.get("solution_code", {})),
                "stdin_wrappers": filter_langs(base_question.get("stdin_wrappers", {})),
                "function_template": filter_langs(base_question.get("function_template", {})),
                # Show first 5 test cases as examples of stdin/stdout format
                "test_cases_sample": base_question.get("test_cases", [])[:5]
            },
            "instructions": """Create a VARIANT of this base question.

🚨 CRITICAL REQUIREMENTS:
1. STDIN FORMAT: Your test_cases.stdin MUST use EXACT SAME FORMAT as test_cases_sample above!
   - If sample uses "1 2 4\\n1 3 4" → you use two lines of space-separated arrays
   - DO NOT add size prefixes or change the format!

2. STDIN_WRAPPERS: Copy the base wrappers and ONLY change function/method names.
   - DO NOT rewrite parsing logic unless absolutely necessary

3. DATA STRUCTURES: Keep SAME data structures as base (e.g., if base uses LinkedList, you use LinkedList)

4. 10 TEST CASES: Exactly 10 (3 visible with is_hidden:false, 7 hidden with is_hidden:true)

5. PLACEHOLDER: Use {user_solution} in all wrappers

6. ALL 3 LANGUAGES: python, java, cpp - must have all

7. ALSO RETURN test_generator_code.python (generate_test_case(size='small')
   returning one random valid stdin string). Expected outputs are computed by
   RUNNING your solution, then cross-checked against an independent oracle
   written from your problem_statement alone - so the statement must be exact."""
        }, indent=2)
        
        try:
            import time
            start_time = time.time()
            
            # GPT-5.4 models use max_completion_tokens instead of max_tokens
            # Need more tokens for 20 test cases + wrappers
            response = await self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": VARIANT_GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=settings.OPENAI_TEMPERATURE
            )
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            content = response.choices[0].message.content
            variant_data = json.loads(content)
            
            # Log concise stats
            usage = response.usage
            tokens_info = f"{usage.prompt_tokens}→{usage.completion_tokens}" if usage else "N/A"
            cost = (usage.prompt_tokens * 0.20 + usage.completion_tokens * 1.25) / 1_000_000 if usage else 0
            
            # Track LLM call metrics
            metrics.record_llm_call(cost=cost, duration_ms=elapsed_ms)
            
            logger.info(f"     LLM response: {elapsed_ms}ms, tokens: {tokens_info}, cost: ${cost:.4f}")
            logger.info(f"     Generated variant: \"{variant_data.get('title', 'N/A')[:40]}\"")
            
            return variant_data
        
        except json.JSONDecodeError as e:
            logger.error(f"     ❌ Failed to parse LLM JSON response: {str(e)[:50]}")
            return None
        except Exception as e:
            logger.error(f"     ❌ LLM generation failed: {str(e)[:80]}")
            return None
    
    async def judge_hardness(self, base_question: dict, variant: dict, facts: dict) -> Optional[dict]:
        """
        Ask the model whether a variant is a meaningfully different question or
        just a relabelling of the base one.

        This is a MODEL JUDGEMENT, not a proof. It is recorded separately from
        the machine-checked evidence and is only used to reject relabellings.
        """
        check = facts.get("base_solution_check", {})
        prompt = f"""Decide whether the VARIANT below is a meaningfully different programming
question from the BASE question, or just the same question relabelled.

BASE QUESTION: {base_question.get('title')} ({base_question.get('difficulty')})
{(base_question.get('problem_statement') or '')[:2500]}

VARIANT: {variant.get('title')}
{(variant.get('problem_statement') or '')[:2500]}

MACHINE-CHECKED FACTS:
- Running the BASE question's own solution against the VARIANT's tests:
  {check.get('passed', 0)} passed, {check.get('failed', 0)} wrong, {check.get('errored', 0)} crashed
- Expected time complexity: {base_question.get('expected_time_complexity')} -> {variant.get('expected_time_complexity')}

COUNT AS "relabelled" (reject) when the solver does essentially the same work:
- renaming the function, variables or the story (robot -> turtle)
- relabelling the symbols or alphabet (digits 1-9 -> 0-8, letters -> numbers)
- flipping an arbitrary convention with no new reasoning (ascending -> descending)
- reformatting input or output only

COUNT AS "meaningful" (accept) when a solver must think differently:
- a new rule or constraint that changes the algorithm or its edge cases
- a different quantity to compute
- tighter limits that force a better time complexity
- combining the base problem with an extra requirement

Return JSON only:
{{"verdict": "meaningful" | "relabelled",
  "harder_than_base": true | false,
  "reason": "one short sentence"}}"""

        try:
            import time
            start_time = time.time()
            response = await self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You judge whether a coding question is genuinely different from another one. Be strict: relabelling is not a new question. Return only JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            elapsed_ms = int((time.time() - start_time) * 1000)
            data = json.loads(response.choices[0].message.content)

            usage = response.usage
            cost = (usage.prompt_tokens * 0.20 + usage.completion_tokens * 1.25) / 1_000_000 if usage else 0
            metrics.record_llm_call(cost=cost, duration_ms=elapsed_ms)

            verdict = str(data.get("verdict", "")).lower()
            if verdict not in ("meaningful", "relabelled"):
                logger.warning(f"     ⚠️ Judge returned an unusable verdict: {verdict!r}")
                return None
            logger.info(f"     LLM judge: {verdict} ({elapsed_ms}ms, ${cost:.4f})")
            return {
                "verdict": verdict,
                "harder_than_base": bool(data.get("harder_than_base")),
                "reason": str(data.get("reason", ""))[:300],
                "note": "model judgement, not a machine-checked proof",
            }
        except json.JSONDecodeError as e:
            logger.error(f"     ❌ Failed to parse judge JSON: {str(e)[:50]}")
            return None
        except Exception as e:
            logger.error(f"     ❌ Hardness judge failed: {str(e)[:80]}")
            return None

    async def write_verification_helpers(self, variant: dict) -> Optional[dict]:
        """
        Write the two helper programs an existing question is missing:
        a brute-force solution and a random-input generator.

        Used to re-verify questions that were stored before verification
        existed - much cheaper than regenerating the whole question.
        """
        solution = (variant.get("solution_code") or {}).get("python", "")
        wrapper = (variant.get("stdin_wrappers") or {}).get("python", "")
        # The oracle must be written from the STATEMENT, not from the reference
        # solution — otherwise it just copies the reference's behaviour and the
        # differential check proves nothing. So only the signature is shared.
        template = (variant.get("function_template") or {}).get("python") or ""
        signature = next(
            (line.strip() for line in (template or solution).splitlines()
             if line.strip().startswith("def ")),
            "def solution(...):",
        )
        sample_inputs = [tc.get("stdin") for tc in (variant.get("test_cases") or [])[:3]]

        prompt = f"""Write two Python helper programs for this existing coding question.

PROBLEM STATEMENT (this is the ONLY definition of correct behaviour):
{variant.get('problem_statement', '')[:4000]}

REQUIRED FUNCTION SIGNATURE (the same wrapper will call it):
```python
{signature}
```

PYTHON STDIN WRAPPER (shows the exact input/output format):
```python
{wrapper}
```

SAMPLE INPUTS (stdin only):
{json.dumps(sample_inputs, indent=2)}

Return JSON with exactly these two fields:

{{"brute_force_solution": {{"python": "..."}},
  "test_generator_code": {{"python": "import random\\n\\ndef generate_test_case(size='small', edge_case=None):\\n    ...\\n    return stdin_string"}}}}

RULES
- brute_force_solution: the SIMPLEST obviously-correct solution (exhaustive search,
  nested loops, plain recursion). It MUST have the SAME function name and
  parameters as the reference solution, because the same wrapper runs it. Write
  it from the problem statement, independently of the reference solution's idea.
  It is an ORACLE, not a submission: it MAY use any operator or built-in the
  problem forbids (e.g. // and % when the statement bans division). It must
  finish fast (under ~0.5s per input) - never loop millions of times, e.g. no
  repeated subtraction when values reach 10^9.
- test_generator_code: generate_test_case(size='small') returns ONE random VALID
  input as a stdin string in EXACTLY the format the wrapper reads. Keep 'small'
  tiny (e.g. n <= 8, small value range) so the brute force finishes instantly.
  Respect every constraint in the statement."""

        try:
            import time
            start_time = time.time()
            response = await self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You write brute-force reference solutions and random input generators for competitive programming questions. Return only JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            elapsed_ms = int((time.time() - start_time) * 1000)
            data = json.loads(response.choices[0].message.content)

            usage = response.usage
            cost = (usage.prompt_tokens * 0.20 + usage.completion_tokens * 1.25) / 1_000_000 if usage else 0
            metrics.record_llm_call(cost=cost, duration_ms=elapsed_ms)

            brute = _python_field(data.get("brute_force_solution"))
            generator = _python_field(data.get("test_generator_code"))
            if not brute or not generator:
                logger.error("     ❌ Helper generation missing brute force or generator")
                return None
            logger.info(f"     LLM helpers: {elapsed_ms}ms, cost: ${cost:.4f}")
            return {
                "brute_force_solution": {"python": brute},
                "test_generator_code": {"python": generator},
            }
        except json.JSONDecodeError as e:
            logger.error(f"     ❌ Failed to parse helper JSON: {str(e)[:50]}")
            return None
        except Exception as e:
            logger.error(f"     ❌ Helper generation failed: {str(e)[:80]}")
            return None

    async def fix_variant(
        self,
        variant: dict,
        failures: list[dict],
        base_question: Optional[dict] = None
    ) -> Optional[dict]:
        """
        Ask LLM to fix a variant based on verification failures.
        
        Args:
            variant: The variant that failed verification
            failures: List of failures with structure:
                {language, index, stdin, expected, actual, error}
            base_question: Original base question for context (problem, working solutions)
        
        Returns:
            Fixed variant dict or None on failure
        """
        failure_description = _describe_failures(failures)
        
        # Build base question context if available
        base_context = ""
        if base_question:
            base_context = f"""
═══════════════════════════════════════════════════════════════════════════════
ORIGINAL BASE QUESTION (for reference - these WORK correctly!):
═══════════════════════════════════════════════════════════════════════════════
Title: {base_question.get('title', 'N/A')}

Sample test stdin format: {base_question.get('test_cases', [{}])[0].get('stdin', 'N/A') if base_question.get('test_cases') else 'N/A'}

WORKING WRAPPERS (use these as reference!):

Python wrapper (WORKS):
```python
{base_question.get('stdin_wrappers', {}).get('python', 'N/A')}
```

Java wrapper (WORKS - notice {{user_solution}} is AFTER Main class!):
```java
{base_question.get('stdin_wrappers', {}).get('java', 'N/A')}
```

C++ wrapper (WORKS):
```cpp
{base_question.get('stdin_wrappers', {}).get('cpp', 'N/A')}
```
"""
        
        fix_prompt = f"""Fix this failing coding problem variant so ALL test cases pass for ALL languages.
{base_context}
PROBLEM VARIANT: {variant.get('title')}

FAILURES:
{''.join(failure_description)}

CURRENT SOLUTION_CODE:
```python
{variant.get('solution_code', {}).get('python', 'MISSING')}
```

```java
{variant.get('solution_code', {}).get('java', 'MISSING')}
```

```cpp
{variant.get('solution_code', {}).get('cpp', 'MISSING')}
```

CURRENT STDIN_WRAPPERS:
Python:
```python
{variant.get('stdin_wrappers', {}).get('python', 'MISSING')}
```

Java:
```java
{variant.get('stdin_wrappers', {}).get('java', 'MISSING')}
```

C++:
```cpp
{variant.get('stdin_wrappers', {}).get('cpp', 'MISSING')}
```

CURRENT BRUTE_FORCE_SOLUTION (python):
```python
{_python_field(variant.get('brute_force_solution')) or 'MISSING'}
```

CURRENT TEST_GENERATOR_CODE (python):
```python
{_python_field(variant.get('test_generator_code')) or 'MISSING'}
```

CURRENT FUNCTION_TEMPLATE (starter code players receive):
{json.dumps(variant.get('function_template', {}), indent=2)}

CURRENT TEST_CASES (first 3, inputs only matter):
{json.dumps(variant.get('test_cases', [])[:3], indent=2)}

ANALYZE: What's the root cause? Which program is wrong according to the problem statement?

Return JSON with ONLY the fields that need fixing:
{{"solution_code": {{"python": "...", "java": "...", "cpp": "..."}}}} - if a reference solution is wrong
{{"stdin_wrappers": {{"python": "...", "java": "...", "cpp": "..."}}}} - if parsing/printing is wrong
{{"brute_force_solution": {{"python": "..."}}}} - if the brute force is wrong
{{"test_generator_code": {{"python": "..."}}}} - if the generator is wrong
{{"function_template": {{"python": "...", "java": "...", "cpp": "..."}}}} - if starter code doesn't fit the wrapper
{{"problem_statement": "..."}} - only to remove an ambiguity

🔴 CRITICAL REMINDERS:
- Java: Use BufferedReader, NOT Scanner (causes timeouts)
- ALL wrappers MUST contain {{user_solution}} placeholder (literal text, not replaced)
- Java wrapper: {{user_solution}} MUST be AFTER the closing brace of Main class"""

        try:
            import time
            start_time = time.time()
            
            response = await self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": FIX_VARIANT_SYSTEM_PROMPT},
                    {"role": "user", "content": fix_prompt}
                ],
                temperature=0.2  # Even lower temp for deterministic fixes
            )
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            content = response.choices[0].message.content
            llm_response = json.loads(content)
            
            usage = response.usage
            tokens_info = f"{usage.prompt_tokens}→{usage.completion_tokens}" if usage else "N/A"
            cost = (usage.prompt_tokens * 0.20 + usage.completion_tokens * 1.25) / 1_000_000 if usage else 0
            
            # Track LLM call metrics
            metrics.record_llm_call(cost=cost, duration_ms=elapsed_ms)
            
            logger.info(f"     LLM fix: {elapsed_ms}ms, tokens: {tokens_info}, cost: ${cost:.4f}")
            
            # CRITICAL: Merge LLM response with original variant
            # LLM might return partial fixes, so we preserve original fields
            fixed_variant = copy.deepcopy(variant)
            
            # Deep merge for nested dicts (solution_code, stdin_wrappers)
            # If LLM only fixes one language, preserve the others
            for key in ["solution_code", "stdin_wrappers", "function_template"]:
                if isinstance(llm_response.get(key), dict) and llm_response[key]:
                    if key not in fixed_variant:
                        fixed_variant[key] = {}
                    # Merge per-language, not replace entire dict
                    for lang, code in llm_response[key].items():
                        if code:  # Only update if provided
                            fixed_variant[key][lang] = code
            
            # Single-language helper programs, stored as {"python": code}
            for key in ["brute_force_solution", "test_generator_code"]:
                code = _python_field(llm_response.get(key))
                if code:
                    fixed_variant[key] = {"python": code}

            if isinstance(llm_response.get("problem_statement"), str) and llm_response["problem_statement"].strip():
                fixed_variant["problem_statement"] = llm_response["problem_statement"]

            # For test_cases - ONLY replace if LLM provides at least 10!
            if "test_cases" in llm_response and llm_response["test_cases"]:
                new_test_count = len(llm_response["test_cases"])
                original_count = len(variant.get("test_cases", []))
                if new_test_count >= 10:
                    fixed_variant["test_cases"] = llm_response["test_cases"]
                    logger.info(f"     Replaced test_cases: {original_count} → {new_test_count}")
                else:
                    # REJECT partial test_cases - LLM tried to reduce them
                    logger.warning(f"     ⚠️ LLM returned only {new_test_count} test_cases (need 10), keeping original {original_count}")
                    # Remove test_cases from llm_response so we don't log it as "fixed"
                    del llm_response["test_cases"]
            
            # Validate required fields and languages
            required_langs = ["python", "java", "cpp"]
            for field in ["solution_code", "stdin_wrappers"]:
                if not fixed_variant.get(field):
                    logger.error(f"     ❌ Fixed variant missing {field}")
                    return None
                for lang in required_langs:
                    if not fixed_variant[field].get(lang):
                        logger.error(f"     ❌ Fixed variant missing {field}.{lang}")
                        return None
            
            if not fixed_variant.get("test_cases"):
                logger.error(f"     ❌ Fixed variant missing test_cases")
                return None
            
            # Log what was changed
            changed = [
                k for k in ["solution_code", "stdin_wrappers", "function_template", "brute_force_solution",
                            "test_generator_code", "test_cases", "problem_statement"]
                if k in llm_response and llm_response[k]
            ]
            logger.info(f"     LLM fixed fields: {changed}")
            
            return fixed_variant
            
        except json.JSONDecodeError as e:
            logger.error(f"     ❌ Failed to parse LLM fix JSON: {str(e)[:50]}")
            return None
        except Exception as e:
            logger.error(f"     ❌ LLM fix failed: {str(e)[:80]}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════════

_openai_instance: Optional[OpenAIClient] = None


def get_openai_client() -> OpenAIClient:
    """Get OpenAI client instance."""
    global _openai_instance
    if _openai_instance is None:
        _openai_instance = OpenAIClient()
    return _openai_instance
