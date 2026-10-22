def recommendations(row, predicted_grade):
    tips=[]
    if row.get("G1",10)<10 or row.get("G2",10)<10:
        tips.append("Review the topics missed in earlier assessments and ask a teacher for targeted feedback.")
    if row.get("studytime",2)<=1:
        tips.append("Set a small, consistent weekly study routine and increase focused practice gradually.")
    if row.get("absences",0)>10:
        tips.append("Catch up on missed lessons and speak with the teacher about any gaps.")
    if row.get("failures",0)>0:
        tips.append("Make a revision plan for foundational topics and schedule regular progress checks.")
    if row.get("schoolsup","no")=="no" and predicted_grade<10:
        tips.append("Consider asking about school support, tutoring, or peer study sessions.")
    if not tips:
        tips.append("Maintain the current study routine and use practice questions to identify topics that need revision.")
    return tips
