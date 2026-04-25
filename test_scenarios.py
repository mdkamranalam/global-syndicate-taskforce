from server.global_syndicate_taskforce_env_environment import GlobalSyndicateTaskforceEnvironment
from models import AMLAction

def run_tests():
    env = GlobalSyndicateTaskforceEnvironment()
    
    print("========================================")
    print("TEST CASE 1: The 'Medium' Phishing Trap")
    print("========================================")
    obs = env.reset(scenario_id="TX-5590")
    print(f"Alert: {obs.system_alerts}")
    
    # Action: Try the WRONG mandate (from the Hard scenario)
    action_wrong = AMLAction(
        target_actor="Bank_Liaison", 
        operation="cross_examine_kyc", 
        policy_mandate="IndiaStack_Section_9" # Wrong mandate for Medium!
    )
    obs = env.step(action_wrong)
    print(f"Action: Used wrong mandate (IndiaStack_Section_9)")
    print(f"Result: {obs.system_alerts} | Reward: {obs.reward:.2f}\n")
    
    # Action: Try the CORRECT mandate for Medium
    action_right = AMLAction(
        target_actor="Bank_Liaison", 
        operation="cross_examine_kyc", 
        policy_mandate="AML_Directive_4"
    )
    obs = env.step(action_right)
    print(f"Action: Used correct mandate (AML_Directive_4)")
    print(f"Result: {obs.system_alerts}")
    print(f"Data Unlocked: {obs.verified_facts} | Reward: {obs.reward:.2f}\n")


    print("========================================")
    print("TEST CASE 2: The 'Easy' Clean Transfer")
    print("========================================")
    obs = env.reset(scenario_id="TX-1024")
    print(f"Alert: {obs.system_alerts}")
    
    # Action: Trust the Analyst and submit early
    action_triage = AMLAction(target_actor="Tier_1_Analyst", operation="fetch_triage_report")
    env.step(action_triage)
    
    action_submit = AMLAction(
        target_actor="Legal_Officer", 
        operation="submit_final_ruling", 
        evidence_chain=["CLEAN_VERIFIED"]
    )
    obs = env.step(action_submit)
    print(f"Action: Submitted 'CLEAN_VERIFIED' based on Analyst.")
    print(f"Result: {obs.system_alerts} | Terminal Reward: {obs.reward:.2f}\n")

if __name__ == "__main__":
    run_tests()