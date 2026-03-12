"""
Tests for rag_knowledge.py - RAG知识库模块
"""
import pytest
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestKnowledgeBaseDefaults:
    """测试RAG未启用时的默认知识返回"""

    def test_default_knowledge_task1_debris(self):
        """测试Task1抛洒物默认知识"""
        from rag_knowledge import (
            _get_default_knowledge_task1,
            _get_default_disposal_task1
        )

        knowledge = _get_default_knowledge_task1("纸箱", "行车道")
        assert "行车道" in knowledge
        assert "纸箱" in knowledge
        assert "抛洒物" in knowledge

        disposal = _get_default_disposal_task1("轮胎", "应急车道")
        assert "轮胎" in disposal
        assert "应急车道" in disposal

    def test_default_knowledge_task3_pavement(self):
        """测试Task3路面病害默认知识"""
        from rag_knowledge import (
            _get_default_knowledge_task3,
            _get_default_disposal_task3
        )

        knowledge = _get_default_knowledge_task3("裂缝", "严重")
        assert "裂缝" in knowledge
        assert "重度" in knowledge

        knowledge_mild = _get_default_knowledge_task3("坑洼", "轻微")
        assert "坑洼" in knowledge_mild
        assert "轻度" in knowledge_mild

        disposal = _get_default_disposal_task3("坑洼", "严重")
        assert "尽快" in disposal or "修复" in disposal

        disposal_mild = _get_default_disposal_task3("裂缝", "轻微")
        assert "适时" in disposal_mild or "养护" in disposal_mild

    def test_default_knowledge_task4_roadside(self):
        """测试Task4路外病害默认知识"""
        from rag_knowledge import (
            _get_default_knowledge_task4,
            _get_default_disposal_task4
        )

        knowledge_slope = _get_default_knowledge_task4("边坡滑坡", "")
        assert "滑坡" in knowledge_slope or "坍塌" in knowledge_slope

        knowledge_water = _get_default_knowledge_task4("排水沟积水", "")
        assert "积水" in knowledge_water or "排水" in knowledge_water

        knowledge_damage = _get_default_knowledge_task4("排水沟破损", "")
        assert "破损" in knowledge_damage or "排水" in knowledge_damage

        disposal_slope = _get_default_disposal_task4("边坡滑坡", "严重")
        assert "边坡" in disposal_slope or "防护" in disposal_slope

        disposal_water = _get_default_disposal_task4("排水沟积水", "中度")
        assert "疏通" in disposal_water or "清理" in disposal_water

    def test_default_knowledge_unknown_type(self):
        """测试未知类型的默认知识"""
        from rag_knowledge import (
            _get_default_knowledge_task1,
            _get_default_knowledge_task4
        )

        knowledge = _get_default_knowledge_task1("未知物体", "行车道")
        assert "未知物体" in knowledge

        knowledge = _get_default_knowledge_task4("未知病害", "")
        assert "安全" in knowledge or "病害" in knowledge


class TestRetrieveKnowledgeFallback:
    """测试RAG不可用时的知识检索回退"""

    @patch('rag_knowledge.get_knowledge_base')
    def test_retrieve_task1_l2_fallback(self, mock_get_kb):
        """测试Task1 L2知识检索回退到默认值"""
        mock_kb = MagicMock()
        mock_kb.is_available.return_value = False
        mock_get_kb.return_value = mock_kb

        from rag_knowledge import retrieve_knowledge_for_task1_l2

        result = retrieve_knowledge_for_task1_l2("纸箱", "行车道")
        assert "纸箱" in result
        assert "行车道" in result

    @patch('rag_knowledge.get_knowledge_base')
    def test_retrieve_task1_l3_fallback(self, mock_get_kb):
        """测试Task1 L3知识检索回退到默认值"""
        mock_kb = MagicMock()
        mock_kb.is_available.return_value = False
        mock_get_kb.return_value = mock_kb

        from rag_knowledge import retrieve_knowledge_for_task1_l3

        result = retrieve_knowledge_for_task1_l3("轮胎", "应急车道", "P1")
        assert "轮胎" in result
        assert "应急车道" in result

    @patch('rag_knowledge.get_knowledge_base')
    def test_retrieve_task3_l2_fallback(self, mock_get_kb):
        """测试Task3 L2知识检索回退到默认值"""
        mock_kb = MagicMock()
        mock_kb.is_available.return_value = False
        mock_get_kb.return_value = mock_kb

        from rag_knowledge import retrieve_knowledge_for_task3_l2

        result = retrieve_knowledge_for_task3_l2("裂缝", "严重")
        assert "裂缝" in result

    @patch('rag_knowledge.get_knowledge_base')
    def test_retrieve_task3_l3_fallback(self, mock_get_kb):
        """测试Task3 L3知识检索回退到默认值"""
        mock_kb = MagicMock()
        mock_kb.is_available.return_value = False
        mock_get_kb.return_value = mock_kb

        from rag_knowledge import retrieve_knowledge_for_task3_l3

        result = retrieve_knowledge_for_task3_l3("坑洼", "严重")
        assert "坑洼" in result

    @patch('rag_knowledge.get_knowledge_base')
    def test_retrieve_task4_l2_fallback(self, mock_get_kb):
        """测试Task4 L2知识检索回退到默认值"""
        mock_kb = MagicMock()
        mock_kb.is_available.return_value = False
        mock_get_kb.return_value = mock_kb

        from rag_knowledge import retrieve_knowledge_for_task4_l2

        result = retrieve_knowledge_for_task4_l2("边坡滑坡")
        assert "滑坡" in result or "边坡" in result

    @patch('rag_knowledge.get_knowledge_base')
    def test_retrieve_task4_l3_fallback(self, mock_get_kb):
        """测试Task4 L3知识检索回退到默认值"""
        mock_kb = MagicMock()
        mock_kb.is_available.return_value = False
        mock_get_kb.return_value = mock_kb

        from rag_knowledge import retrieve_knowledge_for_task4_l3

        result = retrieve_knowledge_for_task4_l3("排水沟积水", "中度")
        assert "排水沟" in result or "疏通" in result


class TestKnowledgeBaseWithRAG:
    """测试RAG启用时的知识检索"""

    @patch('rag_knowledge.get_knowledge_base')
    def test_retrieve_task1_l2_with_rag(self, mock_get_kb):
        """测试Task1 L2知识检索（RAG启用）"""
        mock_kb = MagicMock()
        mock_kb.is_available.return_value = True
        mock_kb.retrieve.side_effect = [
            ["高速公路抛洒物纸箱可能引发交通事故"],
            ["行车道抛洒物应及时清理"],
            []
        ]
        mock_get_kb.return_value = mock_kb

        from rag_knowledge import retrieve_knowledge_for_task1_l2

        result = retrieve_knowledge_for_task1_l2("纸箱", "行车道")
        assert "抛洒物" in result or "纸箱" in result

    @patch('rag_knowledge.get_knowledge_base')
    def test_retrieve_task3_l2_with_rag(self, mock_get_kb):
        """测试Task3 L2知识检索（RAG启用）"""
        mock_kb = MagicMock()
        mock_kb.is_available.return_value = True
        mock_kb.retrieve.side_effect = [
            ["路面裂缝是常见的病害类型"],
            ["裂缝检测标准"],
            []
        ]
        mock_get_kb.return_value = mock_kb

        from rag_knowledge import retrieve_knowledge_for_task3_l2

        result = retrieve_knowledge_for_task3_l2("裂缝", "中度")
        assert "裂缝" in result

    @patch('rag_knowledge.get_knowledge_base')
    def test_retrieve_task4_l2_with_rag(self, mock_get_kb):
        """测试Task4 L2知识检索（RAG启用）"""
        mock_kb = MagicMock()
        mock_kb.is_available.return_value = True
        mock_kb.retrieve.side_effect = [
            ["边坡滑坡是重大安全隐患"],
            ["排水设施检测"],
            []
        ]
        mock_get_kb.return_value = mock_kb

        from rag_knowledge import retrieve_knowledge_for_task4_l2

        result = retrieve_knowledge_for_task4_l2("边坡滑坡")
        assert "边坡" in result or "滑坡" in result


class TestCheckRAGAvailability:
    """测试RAG可用性检查"""

    @patch('rag_knowledge.LANGCHAIN_AVAILABLE', False)
    def test_check_availability_langchain_not_installed(self):
        """测试langchain未安装时的可用性检查"""
        from rag_knowledge import check_rag_availability

        result = check_rag_availability()
        assert result["langchain_available"] is False
        assert result["rag_enabled"] is False
        assert "error" in result

    @patch('rag_knowledge.LANGCHAIN_AVAILABLE', True)
    @patch('os.path.exists')
    def test_check_availability_all_ready(self, mock_exists):
        """测试所有条件满足时的可用性检查"""
        mock_exists.return_value = True

        from rag_knowledge import check_rag_availability

        with patch.dict(os.environ, {
            "EMBEDDING_MODEL_PATH": "/fake/model",
            "KNOWLEDGE_BASE_DIR": "/fake/knowledge"
        }):
            result = check_rag_availability()

        assert result["langchain_available"] is True
        assert result["embedding_model_exists"] is True
        assert result["knowledge_dir_exists"] is True


class TestKnowledgeBaseSingleton:
    """测试KnowledgeBase单例模式"""

    def test_singleton_pattern(self):
        """测试单例模式确保只创建一个实例"""
        from rag_knowledge import KnowledgeBase, _knowledge_base

        # 重置单例
        import rag_knowledge
        rag_knowledge._knowledge_base = None

        # 创建两个实例，应该返回同一个对象
        kb1 = KnowledgeBase(use_rag=False)
        kb2 = KnowledgeBase(use_rag=False)

        assert kb1 is kb2

        # 清理
        rag_knowledge._knowledge_base = None

    @patch('rag_knowledge.LANGCHAIN_AVAILABLE', False)
    def test_get_knowledge_base_singleton(self):
        """测试get_knowledge_base返回单例"""
        import rag_knowledge
        rag_knowledge._knowledge_base = None

        from rag_knowledge import get_knowledge_base

        kb1 = get_knowledge_base(use_rag=False)
        kb2 = get_knowledge_base(use_rag=False)

        assert kb1 is kb2

        # 清理
        rag_knowledge._knowledge_base = None