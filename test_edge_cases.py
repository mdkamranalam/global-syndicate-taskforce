import unittest
from server.global_syndicate_taskforce_env_environment import GlobalSyndicateTaskforceEnvironment
from models import AMLAction

class TestGlobalSyndicateTaskforceEdgeCases(unittest.TestCase):
    def setUp(self):
        self.env = GlobalSyndicateTaskforceEnvironment()
        self.env.reset(scenario_id="TX-5590")  # STOLEN_IDENTITY, needs AML_Directive_4

    def test_infinite_loop_timeout(self):
        """Test that the environment terminates with a penalty after 10 steps to prevent infinite loop hacking."""
        # Spam the analyst 11 times
        for _ in range(10):
            action = AMLAction(target_actor="Tier_1_Analyst", operation="fetch_triage_report")
            obs = self.env.step(action)
            if obs.done:
                break
                
        self.assertTrue(obs.done)
        self.assertIn("Timeout", obs.system_alerts)
        self.assertLess(obs.reward, -0.50) # Step penalties + Timeout penalty

    def test_liaison_bonus_spam_prevention(self):
        """Test that the agent cannot farm +0.30 rewards by repeatedly calling the Bank Liaison."""
        action = AMLAction(
            target_actor="Bank_Liaison", 
            operation="cross_examine_kyc", 
            policy_mandate="AML_Directive_4"
        )
        
        # First call should give the bonus (reward = 0.30 - 0.05 step penalty = 0.25)
        obs1 = self.env.step(action)
        self.assertAlmostEqual(obs1.reward, 0.25)
        
        # Second call should ONLY give the step penalty (reward = -0.05)
        obs2 = self.env.step(action)
        self.assertAlmostEqual(obs2.reward, -0.05)

    def test_invalid_policy_mandate_rejection(self):
        """Test that the Liaison strictly rejects incorrect policy mandates."""
        action = AMLAction(
            target_actor="Bank_Liaison", 
            operation="cross_examine_kyc", 
            policy_mandate="WRONG_MANDATE"
        )
        obs = self.env.step(action)
        self.assertIn("401 Unauthorized", obs.system_alerts)
        self.assertNotIn("STOLEN_IDENTITY", self.env.verified_facts)

    def test_hallucination_trap(self):
        """Test that trusting a hallucination yields a Critical Failure."""
        # We manually submit 'CLEAN_VERIFIED' for TX-5590 (where truth is STOLEN_IDENTITY)
        action = AMLAction(
            target_actor="Legal_Officer", 
            operation="submit_final_ruling", 
            evidence_chain=["CLEAN_VERIFIED"]
        )
        obs = self.env.step(action)
        self.assertTrue(obs.done)
        self.assertIn("Critical Failure", obs.system_alerts)
        self.assertLess(obs.reward, -1.00)

if __name__ == "__main__":
    unittest.main()
