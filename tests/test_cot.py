
def test_cot_reasoning():
    """Test the Chain-of-Thought reasoning flow and parsing."""
    print("\n" + "=" * 60)
    print("Testing CoT Reasoning Layer")
    print("=" * 60)
    
    from minigpt4.models.hero.hero_model import HEROModel
    
    # Check template
    print("\n1. Verifying CoT Template...")
    template = HEROModel.COT_PROMPT_TEMPLATE
    assert "Based on the multimodal evidence" in template
    assert "### Evidence Analysis" in template
    assert "### Rationale" in template
    assert "### Prediction" in template
    print("   ✓ CoT Template is correctly defined.")
    
    # Test Parser
    print("\n2. Testing Output Parser...")
    
    # Mock HEROModel instance for parser testing
    # We don't initialize the full model to avoid loading weights
    class MockHERO:
        def _parse_cot_output(self, text):
            return HEROModel._parse_cot_output(self, text)
            
    mock_model = MockHERO()
    
    sample_output = """
    Based on the evidence...
    
    ### Evidence Analysis
    - Visual: Happy face
    - Audio: Laughter
    
    ### Rationale
    The visual and audio cues matches.
    
    ### Prediction
    {
        "emotion": "happy",
        "confidence": 0.95
    }
    """
    
    parsed = mock_model._parse_cot_output(sample_output)
    print(f"   Input Output:\n{sample_output}")
    print(f"   Parsed Result: {parsed}")
    
    assert parsed['emotion'] == 'happy'
    assert parsed['confidence'] == 0.95
    print("   ✓ Output parser worked correctly.")
    
    # Test Fallback Parser
    print("\n3. Testing Fallback Parser...")
    feature_output = """
    Just returning the JSON directly.
    {
        "emotion": "sad",
        "confidence": 0.8
    }
    """
    parsed_fallback = mock_model._parse_cot_output(feature_output)
    print(f"   Input Output:\n{feature_output}")
    print(f"   Parsed Result: {parsed_fallback}")
    assert parsed_fallback['emotion'] == 'sad'
    print("   ✓ Fallback parser worked correctly.")
    
    # Test Failure Case
    print("\n4. Testing Failure Case...")
    bad_output = "I don't know what to say."
    parsed_bad = mock_model._parse_cot_output(bad_output)
    print(f"   Input Output:\n{bad_output}")
    print(f"   Parsed Result: {parsed_bad}")
    assert parsed_bad['emotion'] == 'unknown'
    print("   ✓ Failure handling worked correctly.")

    print("\n   ✓ CoT Reasoning Layer test passed!")
    return True

# Add to run_all_tests
