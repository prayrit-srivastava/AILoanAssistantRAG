from rag import ask_loan_assistant

question = """
Customer wants a Home Loan.

Customer has poor credit score.

Suggest record type,
documents,
next steps.
"""

print(
    ask_loan_assistant(question)
)