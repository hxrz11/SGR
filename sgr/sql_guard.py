from __future__ import annotations

from typing import Iterable, Optional, Set
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

# Разрешены только SELECT-запросы. НЕТ авто-LIMIT.


@dataclass
class GuardConfig:
    allowed_schemas: Optional[Set[str]] = None  # None = любая
    allowed_tables: Optional[Set[str]] = None
    allowed_columns: Optional[Set[str]] = None
    allowed_functions: Optional[Set[str]] = None
    deny_comments: bool = True
    deny_multiple_statements: bool = True
    deny_cte: bool = False  # можно включить при необходимости
    deny_set_ops: bool = True  # UNION/INTERSECT/EXCEPT
    deny_subqueries: bool = False


class SQLGuardError(Exception):
    pass


class SQLGuard:
    def __init__(self, cfg: GuardConfig):
        self.cfg = cfg

    def _assert(self, cond: bool, msg: str):
        if not cond:
            raise SQLGuardError(msg)

    def validate_text_level(self, sql: str):
        # Комментарии и «стэкинг» операторов
        if self.cfg.deny_comments:
            self._assert("--" not in sql and "/*" not in sql and "*/" not in sql, "Comments are not allowed")
        if self.cfg.deny_multiple_statements:
            self._assert(";" not in sql.strip().rstrip(";"), "Multiple statements are not allowed")

    def _walk(self, node: exp.Expression) -> Iterable[exp.Expression]:
        yield node
        for child in node.iter_expressions():
            yield from self._walk(child)

    def validate_ast(self, sql: str):
        try:
            parsed = sqlglot.parse_one(sql, read="postgres")
        except Exception as e:
            raise SQLGuardError(f"SQL parse error: {e}")

        # Разрешен только SELECT
        self._assert(
            isinstance(parsed, exp.Select)
            or isinstance(parsed, exp.Subquery)
            or isinstance(parsed, exp.With),
            "Only SELECT queries are allowed",
        )

        # CTE
        if isinstance(parsed, exp.With):
            self._assert(not self.cfg.deny_cte, "CTE (WITH) is not allowed by policy")
            root = parsed.this
        else:
            root = parsed

        # Операции над множествами (UNION/EXCEPT/INTERSECT)
        if self.cfg.deny_set_ops:
            for n in self._walk(root):
                self._assert(
                    not isinstance(n, (exp.Union, exp.Except, exp.Intersect)),
                    "Set operations are not allowed",
                )

        # Таблицы/схемы
        for table in root.find_all(exp.Table):
            if table.this:
                tbl = table.this.name
                sch = table.db.name if table.db else None
                if self.cfg.allowed_tables is not None:
                    self._assert(tbl in self.cfg.allowed_tables, f"Table not allowed: {tbl}")
                if self.cfg.allowed_schemas is not None and sch is not None:
                    self._assert(sch in self.cfg.allowed_schemas, f"Schema not allowed: {sch}")

        # Колонки
        if self.cfg.allowed_columns is not None:
            for col in root.find_all(exp.Column):
                name = col.name
                # пропускаем звёздочку только если явно разрешена
                if name == "*":
                    self._assert("*" in self.cfg.allowed_columns, "Wildcard * is not allowed")
                else:
                    self._assert(name in self.cfg.allowed_columns, f"Column not allowed: {name}")

        # Функции
        if self.cfg.allowed_functions is not None:
            for func in root.find_all(exp.Func):
                fname = func.name.upper() if func.name else ""
                self._assert(
                    fname in {f.upper() for f in self.cfg.allowed_functions},
                    f"Function not allowed: {fname}",
                )

        # Подзапросы
        if self.cfg.deny_subqueries:
            for sub in root.find_all(exp.Subquery):
                # разрешаем субзапрос только если он верхний уровень
                raise SQLGuardError("Subqueries are not allowed by policy")

    def validate(self, sql: str):
        self.validate_text_level(sql)
        self.validate_ast(sql)
        return True
