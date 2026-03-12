"""
Tests for prompts.py
"""
import pytest
from prompts import (
    TASK1_SYSTEM_PROMPT,
    TASK1_USER_PROMPT,
    TASK2_SYSTEM_PROMPT,
    TASK2_USER_PROMPT,
    TASK3_SYSTEM_PROMPT,
    TASK3_USER_PROMPT,
    TASK4_SYSTEM_PROMPT,
    TASK4_USER_PROMPT,
    get_prompts_by_scene,
    get_prompts_by_task,
)


class TestPromptConstants:
    """Tests for prompt constants"""

    def test_task1_prompts_exist(self):
        """Test Task1 prompts exist"""
        assert TASK1_SYSTEM_PROMPT is not None
        assert TASK1_USER_PROMPT is not None
        assert len(TASK1_SYSTEM_PROMPT) > 0
        assert len(TASK1_USER_PROMPT) > 0

    def test_task2_prompts_exist(self):
        """Test Task2 prompts exist"""
        assert TASK2_SYSTEM_PROMPT is not None
        assert TASK2_USER_PROMPT is not None
        assert len(TASK2_SYSTEM_PROMPT) > 0
        assert len(TASK2_USER_PROMPT) > 0

    def test_task3_prompts_exist(self):
        """Test Task3 prompts exist"""
        assert TASK3_SYSTEM_PROMPT is not None
        assert TASK3_USER_PROMPT is not None
        assert len(TASK3_SYSTEM_PROMPT) > 0
        assert len(TASK3_USER_PROMPT) > 0

    def test_task4_prompts_exist(self):
        """Test Task4 prompts exist"""
        assert TASK4_SYSTEM_PROMPT is not None
        assert TASK4_USER_PROMPT is not None
        assert len(TASK4_SYSTEM_PROMPT) > 0
        assert len(TASK4_USER_PROMPT) > 0

    def test_task1_system_prompt_content(self):
        """Test Task1 system prompt contains key terms"""
        assert "抛洒物" in TASK1_SYSTEM_PROMPT
        assert "高速公路" in TASK1_SYSTEM_PROMPT
        assert "JSON" in TASK1_SYSTEM_PROMPT

    def test_task2_system_prompt_content(self):
        """Test Task2 system prompt contains key terms"""
        assert "违停" in TASK2_SYSTEM_PROMPT
        assert "双闪" in TASK2_SYSTEM_PROMPT or "危险报警闪光灯" in TASK2_SYSTEM_PROMPT
        # JSON is not explicitly in the prompt, check for format-related terms
        assert "输出" in TASK2_SYSTEM_PROMPT or "格式" in TASK2_SYSTEM_PROMPT

    def test_task3_system_prompt_content(self):
        """Test Task3 system prompt contains key terms"""
        assert "病害" in TASK3_SYSTEM_PROMPT
        assert "裂缝" in TASK3_SYSTEM_PROMPT
        assert "护栏" in TASK3_SYSTEM_PROMPT

    def test_task4_system_prompt_content(self):
        """Test Task4 system prompt contains key terms"""
        assert "边坡" in TASK4_SYSTEM_PROMPT
        assert "排水沟" in TASK4_SYSTEM_PROMPT


class TestGetPromptsByScene:
    """Tests for get_prompts_by_scene function"""

    def test_scene_0_task1(self):
        """Test scene_id 0 returns Task1 prompts"""
        system, user = get_prompts_by_scene(0)
        assert system == TASK1_SYSTEM_PROMPT
        assert user == TASK1_USER_PROMPT

    def test_scene_1_task2(self):
        """Test scene_id 1 returns Task2 prompts"""
        system, user = get_prompts_by_scene(1)
        assert system == TASK2_SYSTEM_PROMPT
        assert user == TASK2_USER_PROMPT

    def test_scene_2_task3(self):
        """Test scene_id 2 returns Task3 prompts"""
        system, user = get_prompts_by_scene(2)
        assert system == TASK3_SYSTEM_PROMPT
        assert user == TASK3_USER_PROMPT

    def test_scene_3_task3(self):
        """Test scene_id 3 returns Task3 prompts"""
        system, user = get_prompts_by_scene(3)
        assert system == TASK3_SYSTEM_PROMPT
        assert user == TASK3_USER_PROMPT

    def test_scene_4_task3(self):
        """Test scene_id 4 returns Task3 prompts"""
        system, user = get_prompts_by_scene(4)
        assert system == TASK3_SYSTEM_PROMPT
        assert user == TASK3_USER_PROMPT

    def test_scene_5_task3(self):
        """Test scene_id 5 returns Task3 prompts"""
        system, user = get_prompts_by_scene(5)
        assert system == TASK3_SYSTEM_PROMPT
        assert user == TASK3_USER_PROMPT

    def test_scene_6_task4(self):
        """Test scene_id 6 returns Task4 prompts"""
        system, user = get_prompts_by_scene(6)
        assert system == TASK4_SYSTEM_PROMPT
        assert user == TASK4_USER_PROMPT

    def test_scene_7_task4(self):
        """Test scene_id 7 returns Task4 prompts"""
        system, user = get_prompts_by_scene(7)
        assert system == TASK4_SYSTEM_PROMPT
        assert user == TASK4_USER_PROMPT

    def test_scene_8_task4(self):
        """Test scene_id 8 returns Task4 prompts"""
        system, user = get_prompts_by_scene(8)
        assert system == TASK4_SYSTEM_PROMPT
        assert user == TASK4_USER_PROMPT

    def test_scene_unknown_raises(self):
        """Test unknown scene_id raises ValueError"""
        with pytest.raises(ValueError):
            get_prompts_by_scene(99)


class TestGetPromptsByTask:
    """Tests for get_prompts_by_task function"""

    def test_task_1(self):
        """Test task_id 1 returns Task1 prompts"""
        system, user = get_prompts_by_task(1)
        assert system == TASK1_SYSTEM_PROMPT
        assert user == TASK1_USER_PROMPT

    def test_task_2(self):
        """Test task_id 2 returns Task2 prompts"""
        system, user = get_prompts_by_task(2)
        assert system == TASK2_SYSTEM_PROMPT
        assert user == TASK2_USER_PROMPT

    def test_task_3(self):
        """Test task_id 3 returns Task3 prompts"""
        system, user = get_prompts_by_task(3)
        assert system == TASK3_SYSTEM_PROMPT
        assert user == TASK3_USER_PROMPT

    def test_task_4(self):
        """Test task_id 4 returns Task4 prompts"""
        system, user = get_prompts_by_task(4)
        assert system == TASK4_SYSTEM_PROMPT
        assert user == TASK4_USER_PROMPT

    def test_task_unknown_raises(self):
        """Test unknown task_id raises ValueError"""
        with pytest.raises(ValueError):
            get_prompts_by_task(99)


class TestPromptJsonFormat:
    """Tests for JSON format in prompts"""

    def test_task1_user_prompt_json_format(self):
        """Test Task1 user prompt has JSON format example"""
        assert "```json" in TASK1_USER_PROMPT
        assert "l1_result" in TASK1_USER_PROMPT
        assert "bbox" in TASK1_USER_PROMPT

    def test_task2_user_prompt_json_format(self):
        """Test Task2 user prompt has JSON format example"""
        assert "```json" in TASK2_USER_PROMPT
        assert "l1_result" in TASK2_USER_PROMPT
        assert "has_violation" in TASK2_USER_PROMPT

    def test_task3_user_prompt_json_format(self):
        """Test Task3 user prompt has JSON format example"""
        assert "```json" in TASK3_USER_PROMPT
        assert "l1_result" in TASK3_USER_PROMPT
        assert "scene_id" in TASK3_USER_PROMPT

    def test_task4_user_prompt_json_format(self):
        """Test Task4 user prompt has JSON format example"""
        assert "```json" in TASK4_USER_PROMPT
        assert "l1_result" in TASK4_USER_PROMPT
        assert "scene_id" in TASK4_USER_PROMPT