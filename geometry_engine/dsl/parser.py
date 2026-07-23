"""Parser for the first Geometry Engine DSL."""

from __future__ import annotations

import ast
from typing import Any, List

from geometry_engine.dsl.command import ParsedCommand


class DSLParseError(ValueError):
    """Raised when DSL text cannot be parsed safely."""


class DSLParser:
    """Parse function-call-style DSL commands into ParsedCommand objects."""

    def parse(self, source: str) -> ParsedCommand:
        """Parse one DSL command."""

        commands = self.parse_script(source)
        if len(commands) != 1:
            raise DSLParseError(f"Expected exactly one command, got {len(commands)}.")
        return commands[0]

    def parse_script(self, source: str) -> List[ParsedCommand]:
        """Parse one or more DSL commands from source text."""

        chunks = self._split_commands(source)
        return [self._parse_call(chunk) for chunk in chunks]

    def _parse_call(self, source: str) -> ParsedCommand:
        """Parse one function-call expression."""

        try:
            expression = ast.parse(source, mode="eval")
        except SyntaxError as exc:
            raise DSLParseError(f"Invalid DSL syntax: {source!r}") from exc

        call = expression.body
        if not isinstance(call, ast.Call):
            raise DSLParseError("DSL command must be a function-style call.")
        if not isinstance(call.func, ast.Name):
            raise DSLParseError("DSL command name must be a simple identifier.")

        args = [self._literal(arg) for arg in call.args]
        kwargs = {}
        for keyword in call.keywords:
            if keyword.arg is None:
                raise DSLParseError("Variadic keyword arguments are not supported.")
            kwargs[keyword.arg] = self._literal(keyword.value)
        return ParsedCommand(name=call.func.id, args=args, kwargs=kwargs)

    def _literal(self, node: ast.AST) -> Any:
        """Return a safe literal value from an AST node."""

        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Str):
            return node.s
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.NameConstant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self._literal(node.operand)
            if isinstance(value, (int, float)):
                return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.List):
            return [self._literal(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self._literal(item) for item in node.elts)
        raise DSLParseError(f"Only literal DSL arguments are supported, got {ast.dump(node)}.")

    @staticmethod
    def _split_commands(source: str) -> List[str]:
        """Split a script into top-level command chunks."""

        chunks: List[str] = []
        buffer: List[str] = []
        depth = 0
        quote: str | None = None
        escape = False

        for char in source:
            if quote is not None:
                buffer.append(char)
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == quote:
                    quote = None
                continue

            if char in {"'", '"'}:
                quote = char
                buffer.append(char)
            elif char == "(":
                depth += 1
                buffer.append(char)
            elif char == ")":
                depth -= 1
                if depth < 0:
                    raise DSLParseError("Unbalanced closing parenthesis.")
                buffer.append(char)
            elif char in {"\n", ";"} and depth == 0:
                chunk = "".join(buffer).strip()
                if chunk:
                    chunks.append(chunk)
                buffer = []
            else:
                buffer.append(char)

        if quote is not None or depth != 0:
            raise DSLParseError("Unbalanced quote or parenthesis in DSL source.")
        chunk = "".join(buffer).strip()
        if chunk:
            chunks.append(chunk)
        return chunks
