"""Tests for the ORM model → SQL table linker (roadmap C4)."""

import networkx as nx

from plugin_implementation.code_graph.orm_linker import (
    _to_snake_case,
    _detect_orm,
    _candidate_table_names,
    link_orm_models,
)


def _add_table(g, name, node_id=None):
    nid = node_id or f"sql::schema.sql::sql_table:{name}"
    g.add_node(
        nid,
        name=name,
        symbol_name=name,
        type="sql_table",
        symbol_type="sql_table",
        language="sql",
    )
    return nid


def _add_class(g, name, source_text, language, node_id=None):
    nid = node_id or f"{language}::models.py::{name}"
    g.add_node(
        nid,
        name=name,
        symbol_name=name,
        type="class",
        symbol_type="class",
        language=language,
        source_text=source_text,
    )
    return nid


class TestSnakeCase:
    def test_camel_to_snake(self):
        assert _to_snake_case("BlogPost") == "blog_post"

    def test_single_word(self):
        assert _to_snake_case("User") == "user"

    def test_acronym_run(self):
        assert _to_snake_case("APIKey") == "a_p_i_key"


class TestDetectOrm:
    def test_sqlalchemy_column(self):
        assert _detect_orm("id = Column(Integer, primary_key=True)", "python")

    def test_sqlalchemy_tablename(self):
        assert _detect_orm("__tablename__ = 'users'", "python")

    def test_django_field(self):
        assert _detect_orm("name = models.CharField(max_length=100)", "python")

    def test_hibernate_entity(self):
        assert _detect_orm("@Entity\npublic class User {}", "java")

    def test_plain_class_not_orm(self):
        assert not _detect_orm("def hello(): pass", "python")

    def test_unsupported_language(self):
        assert not _detect_orm("Column(Integer)", "rust")


class TestCandidateTableNames:
    def test_sqlalchemy_explicit_tablename(self):
        names = _candidate_table_names(
            "User", "__tablename__ = 'app_users'", "python"
        )
        assert names[0] == "app_users"

    def test_django_db_table(self):
        names = _candidate_table_names(
            "BlogPost", "class Meta:\n    db_table = 'posts'", "python"
        )
        assert names[0] == "posts"

    def test_python_default_snake_case(self):
        names = _candidate_table_names("BlogPost", "id = Column(Integer)", "python")
        assert "blog_post" in [n.lower() for n in names]

    def test_hibernate_table_annotation(self):
        names = _candidate_table_names(
            "User", "@Entity\n@Table(name=\"app_user\")\nclass User {}", "java"
        )
        assert names[0] == "app_user"


class TestLinkOrmModels:
    def test_no_tables_is_noop(self):
        g = nx.MultiDiGraph()
        _add_class(g, "User", "id = Column(Integer)", "python")
        assert link_orm_models(g) == 0

    def test_sqlalchemy_explicit_match(self):
        g = nx.MultiDiGraph()
        tid = _add_table(g, "users")
        cid = _add_class(
            g, "User",
            "__tablename__ = 'users'\n    id = Column(Integer, primary_key=True)",
            "python",
        )
        assert link_orm_models(g) == 1
        assert g.has_edge(cid, tid, key="models_table")
        data = g.get_edge_data(cid, tid, key="models_table")
        assert data["edge_class"] == "cross_language"
        assert data["relationship_type"] == "models_table"

    def test_django_snake_case_match(self):
        g = nx.MultiDiGraph()
        tid = _add_table(g, "blog_post")
        cid = _add_class(
            g, "BlogPost", "title = models.CharField(max_length=80)", "python"
        )
        assert link_orm_models(g) == 1
        assert g.has_edge(cid, tid, key="models_table")

    def test_hibernate_entity_match(self):
        g = nx.MultiDiGraph()
        tid = _add_table(g, "app_user")
        cid = _add_class(
            g, "User",
            "@Entity\n@Table(name=\"app_user\")\npublic class User {}",
            "java",
        )
        assert link_orm_models(g) == 1
        assert g.has_edge(cid, tid, key="models_table")

    def test_no_match_when_table_name_differs(self):
        g = nx.MultiDiGraph()
        _add_table(g, "accounts")
        _add_class(g, "User", "__tablename__ = 'users'", "python")
        assert link_orm_models(g) == 0

    def test_plain_class_ignored(self):
        g = nx.MultiDiGraph()
        _add_table(g, "users")
        _add_class(g, "UserService", "def get(self): pass", "python")
        assert link_orm_models(g) == 0

    def test_idempotent(self):
        g = nx.MultiDiGraph()
        _add_table(g, "users")
        _add_class(g, "User", "__tablename__ = 'users'", "python")
        assert link_orm_models(g) == 1
        # Second run adds nothing (edge already present).
        assert link_orm_models(g) == 0
