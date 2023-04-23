import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Load the training data into a pandas dataframe
class intent: 

  def __init__(self):
        self.train_data= {'input': ["What are some easy ways to save energy at home?"
          ,"How can I reduce my electricity bill?"
          ,"Can you suggest some energy-saving tips for the winter season?"
          ,"What are the most effective ways to reduce energy consumption in the workplace?"
          ,"Can you recommend some energy-efficient appliances to save on electricity costs?"
          ,"What are some practical tips for saving energy in the kitchen?"
          ,"What are some common energy-wasting habits to avoid?"
          ,"Can you provide some tips for reducing energy usage while doing laundry?"
          ,"What are some ways to conserve water and energy at the same time?"
          ,"What are some effective ways to keep my home cool in the summer without using too much energy?"
          ,"Can you suggest some energy-saving tips for those who work from home?"
          ,"What are some energy-saving tips for those who live in apartments or small spaces?"
          ,"How can I make my home more energy-efficient without spending a lot of money?"
          ,"Can you suggest some DIY projects to improve energy efficiency in my home?"
          ,"What are some effective ways to reduce energy usage while using my computer or other electronic devices?"
          ,"Can you provide some tips for saving energy during the holiday season?"
          ,"What are some effective ways to save energy while traveling?"
          ,"What are some tips for reducing energy usage in my backyard or outdoor spaces?"
          ,"Can you recommend some energy-saving tips for parents with young children?"
          ,"What are some ways to reduce energy usage and save money while using heating and cooling systems?"
          ,"How can I make my home more energy-efficient using smart home automation?"
          ,"Can you recommend some smart home automation devices that can improve home security?"
          ,"What are some smart home automation products that can make my life easier?"
          ,"What are some affordable options for smart home automation?"
          ,"What are some effective ways to control lighting using smart home automation?"
          ,"Can you suggest some smart home automation solutions for those with disabilities or mobility issues?"
          ,"What are the most popular voice assistants for smart home automation?"
          ,"What are some smart home automation systems that can help me save money on utility bills?"
          ,"How can I control my thermostat using smart home automation?"
          ,"What are some smart home automation products that can help with home entertainment?"
          ,"Using smart home automation, you can easily control your home appliances and save time and energy."
          ,"Smart home automation can improve the air quality in your home with air purifiers and other devices."
          ,"Here are some smart home automation systems that can help you save money on home insurance."
          ,"Smart home automation can help you create a safer home environment with integrated security systems."
          ,"Using smart home automation, you can easily monitor and control your home's energy usage."
          ,"These are some smart home automation solutions that can improve the quality of life for pet owners."
          ,"Smart home automation can integrate with your home's existing systems, such as your HVAC or irrigation systems."
          ,"These are some popular smart home automation brands that offer a wide range of products and services."
          ,"Smart home automation can help those with allergies or sensitivities create a healthier home environment."
          ,"Using smart home automation, you can easily control and customize your home's lighting and ambiance."
          ,"Here are some workplace energy-saving tips that can help your business save money on utility bills."
          ,"Investing in energy-efficient office equipment and lighting can help reduce your workplace's carbon footprint."
          ,"Workplace energy-saving programs can not only help save the environment but also boost employee morale and productivity."
          ,"These are some of the best energy-saving solutions for small businesses and startups."
          ,"Installing occupancy sensors and timers can help save energy and reduce waste in your workplace."
          ,"Using natural light and creating an open office layout can reduce the need for artificial lighting and save energy."
          ,"Implementing a remote work policy can help reduce energy usage and transportation-related emissions."
          ,"Training employees on energy-saving practices can help create a culture of sustainability in the workplace."
          ,"Replacing outdated heating, ventilation, and air conditioning (HVAC) systems with energy-efficient alternatives can help save energy and reduce costs."
          ,"Green building certifications like LEED can help your workplace become more sustainable and energy-efficient."
          ,"What are some strategies for reducing the need for artificial lighting in the workplace and saving energy?"
          ,"How can implementing a remote work policy help reduce energy usage and transportation-related emissions?"
          ,"What are some effective ways to train employees on energy-saving practices in the workplace?"
          ,"How can replacing outdated HVAC systems with energy-efficient alternatives help save energy and reduce costs in the workplace?"
          ,"What are some green building certifications like LEED that can help my workplace become more sustainable and energy-efficient?"
          ,"What are some effective maintenance strategies for improving energy efficiency and prolonging the lifespan of appliances in the workplace?"
          ,"How can power strips and unplugging electronics when not in use help reduce standby power consumption in the workplace?"
          ,"What are some recycling and composting programs that can help reduce waste and save energy in the workplace?"
          ,"How can investing in renewable energy sources like solar panels help power my workplace and save money on"
          ,"What are the best energy-saving solutions for small businesses and startups?"
          ,"Here are some examples of energy-saving technology that you can implement in your home or office."
          ,"Energy-saving technology is becoming increasingly popular as people look for ways to reduce their carbon footprint."
          ,"These are some of the best energy-saving technologies available on the market."
          ,"By implementing energy-saving technology, you can significantly reduce your energy bills and save money."
          ,"Energy-saving technology can help you achieve a more sustainable lifestyle and reduce your impact on the environment."
          ,"Here are some of the most innovative energy-saving technologies that are currently available."
          ,"Energy-saving technology can be easily integrated into your existing appliances and devices for maximum efficiency."
          ,"These are some affordable options for those interested in implementing energy-saving technology in their home or office."
          ,"Energy-saving technology can help you reduce your carbon footprint and contribute to a more sustainable future."
          ,"By using energy-saving technology, you can improve the energy efficiency of your home or office and reduce your environmental impact."
          ,"Here are some energy-saving technologies that can help you reduce your water consumption and waste less."
          ,"Energy-saving technology can help you monitor your energy usage and identify areas where you can make improvements."
          ,"These are some energy-saving technologies that can help you reduce your reliance on fossil fuels and transition to renewable energy sources."
          ,"Energy-saving technology can help you optimize your energy usage for maximum efficiency and cost savings."
          ,"Here are some energy-saving technologies that can help you reduce your HVAC costs and improve indoor air quality."
          ,"Energy-saving technology can be customized to meet the unique needs and requirements of your home or office."
          ,"These are some energy-saving technologies that can help you achieve a more sustainable lifestyle without sacrificing comfort or convenience."
          ,"Energy-saving technology can help you reduce your environmental impact and promote a more sustainable future for all."
          ,"Here are some energy-saving technologies that can help you reduce your overall energy consumption and minimize waste."
          ,"Energy-saving technology can be easily integrated into your daily routine to help you live a more sustainable lifestyle."
          ,"Use this energy-saving checklist to reduce your carbon footprint and lower your energy bills."
          ,"Our energy-saving checklist can help you make your home more efficient and comfortable."
          ,"Here is a step-by-step energy-saving checklist to help you save money and protect the environment."
          ,"With our energy-saving checklist, you can easily identify areas where you can improve your energy efficiency."
          ,"This energy-saving checklist covers everything from lighting to appliances to help you save energy."
          ,"Use this energy-saving checklist to evaluate your home's energy usage and find areas for improvement."
          ,"Our energy-saving checklist can help you prioritize your energy-saving efforts and make the most impact."
          ,"This energy-saving checklist includes tips for both homeowners and renters to reduce their energy consumption."
          ,"By following our energy-saving checklist, you can save up to 20 on your energy bills each year."
          ,"Use this energy-saving checklist as a guide to make your home more energy-efficient and comfortable."
          ,"What are some items on an energy-saving checklist?"
          ,"Can you provide a sample energy-saving checklist?"
          ,"How do I use an energy-saving checklist to make my home more efficient?"
          ,"Are there any energy-saving checklists specifically for apartment dwellers?"
          ,"What is the benefit of using an energy-saving checklist?"
          ,"Can an energy-saving checklist help me save money on my utility bills?"
          ,"What types of energy-saving measures are typically included on an energy-saving checklist?"
          ,"How often should I refer to an energy-saving checklist to maintain energy efficiency in my home?"
          ,"Is there a recommended order for completing the items on an energy-saving checklist?"
          ,"Are there any apps or tools available to help me complete an energy-saving checklist?"
          ,"How much money can I save by implementing energy-saving measures?"
          ,"What are the upfront costs associated with energy-saving upgrades?"
          ,"What kind of return on investment can I expect from energy-saving upgrades?"
          ,"Can you provide examples of energy-saving upgrades that have the quickest payback periods?"
          ,"What are some low-cost energy-saving options that I can implement right away?"
          ,"Are there any government incentives or rebates available to offset the costs of energy-saving upgrades?"
          ,"What is the average cost savings for homeowners who have implemented energy-saving upgrades?"
          ,"Are there any financing options available to help me pay for energy-saving upgrades?"
          ,"How can I estimate the costs and savings of energy-saving upgrades for my specific home?"
          ,"What are some factors that can affect the costs and savings of energy-saving upgrades?"
          ,"Implementing energy-saving measures can lead to significant cost savings over time."
          ,"By investing in energy-saving upgrades, you can reduce your monthly utility bills."
          ,"Energy-saving upgrades may require an initial investment, but they can pay for themselves in the long run."
          ,"There are a variety of energy-saving options available at different price points to fit any budget."
          ,"Upgrading to energy-efficient appliances can result in substantial cost savings over their lifespan."
          ,"Investing in energy-saving upgrades can increase the value of your home while reducing your energy bills."
          ,"The cost of energy-saving upgrades can vary depending on the size and age of your home."
          ,"Using energy-saving practices can help reduce your carbon footprint and save you money at the same time."
          ,"By reducing your energy consumption, you can help protect the environment while also saving money."
          ,"Energy-saving upgrades may require some upfront costs, but they can lead to long-term cost savings and environmental benefits."
          ,"What are some financing options available for energy-saving upgrades?"
          ,"Can you provide information on low-interest loans for energy-saving upgrades?"
          ,"Are there any government programs or incentives that provide financial assistance for energy-saving upgrades?"
          ,"What is the process for applying for financing for energy-saving upgrades?"
          ,"Can I use a home equity loan to finance energy-saving upgrades?"
          ,"What are some of the most cost-effective financing options for energy-saving upgrades?"
          ,"Are there any grants or rebates available for energy-saving upgrades?"
          ,"How can I determine the best financing option for my specific energy-saving project?"
          ,"What are some of the pros and cons of different financing options for energy-saving upgrades?"
          ,"What kind of credit score is required to qualify for financing for energy-saving upgrades?"
          ,"Financing energy-saving upgrades can help you reduce your monthly utility bills and save money in the long run."
          ,"Investing in energy-saving upgrades can increase the value of your home and provide long-term financial benefits."
          ,"There are a variety of financing options available to help you pay for energy-saving upgrades."
          ,"Using financing to pay for energy-saving upgrades can help you avoid upfront costs while still reaping the benefits."
          ,"Financing energy-saving upgrades can be a smart investment in the future of your home and the environment."
          ,"By financing energy-saving upgrades, you can reduce your carbon footprint and contribute to a more sustainable future."
          ,"Financing options for energy-saving upgrades can vary depending on your location and specific project needs."
          ,"Financing energy-saving upgrades can be a practical way to improve the energy efficiency of your home without breaking the bank."
          ,"Using financing to pay for energy-saving upgrades can help you take control of your energy consumption and reduce your environmental impact."
          ,"Investing in energy-saving upgrades with financing can be a wise financial decision that benefits both your wallet and the planet."
          ,"What are some common energy-saving goals for homeowners and businesses?"
          ,"How can I set realistic energy-saving goals for my home or workplace?"
          ,"What are some strategies for achieving energy-saving goals on a tight budget?"
          ,"What kind of savings can I expect to see by setting and achieving energy-saving goals?"
          ,"What are some common obstacles that prevent people from achieving their energy-saving goals?"
          ,"What are some tools or resources that can help me track my progress toward my energy-saving goals?"
          ,"How can I encourage my family or coworkers to get on board with our energy-saving goals?"
          ,"Are there any energy-saving goals that are specific to certain types of buildings or industries?"
          ,"What are some long-term benefits of achieving energy-saving goals, beyond just saving money on utilities?"
          ,"How often should I reassess and adjust my energy-saving goals to make sure I'm staying on track?"
          ,"Setting and achieving energy-saving goals can help you reduce your carbon footprint and contribute to a more sustainable future."
          ,"Energy-saving goals can be an effective way to manage your energy consumption and save money on utilities."
          ,"By setting realistic energy-saving goals, you can make a meaningful impact on your monthly energy bills."
          ,"Setting energy-saving goals for your home or business can help you identify areas where you can make improvements and reduce waste."
          ,"Achieving your energy-saving goals can be a satisfying accomplishment that also benefits the environment."
          ,"Having clear energy-saving goals can help you prioritize your investments in energy-efficient upgrades and improvements."
          ,"Energy-saving goals can be customized to meet the specific needs and priorities of your home or workplace."
          ,"With the right strategies and tools, achieving your energy-saving goals can be an achievable and rewarding process."
          ,"Regularly reassessing and adjusting your energy-saving goals can help you stay motivated and make consistent progress over time."
          ,"By setting and achieving energy-saving goals, you can make a positive impact on the environment and inspire others to do the same.",]
          ,'intent':["ENERGY_SAVING_TIPS","ENERGY_SAVING_TIPS","ENERGY_SAVING_TIPS","ENERGY_SAVING_TIPS","ENERGY_SAVING_TIPS","ENERGY_SAVING_TIPS","ENERGY_SAVING_TIPS","ENERGY_SAVING_TIPS","ENERGY_SAVING_TIPS","ENERGY_SAVING_TIPS","ENERGY_SAVING_TIPS","ENERGY_SAVING_TIPS","ENERGY_SAVING_TIPS","ENERGY_SAVING_TIPS","ENERGY_SAVING_TIPS","ENERGY_SAVING_TIPS","ENERGY_SAVING_TIPS","ENERGY_SAVING_TIPS","ENERGY_SAVING_TIPS","ENERGY_SAVING_TIPS",
                "SMART_HOME_AUTOMATION","SMART_HOME_AUTOMATION","SMART_HOME_AUTOMATION","SMART_HOME_AUTOMATION","SMART_HOME_AUTOMATION","SMART_HOME_AUTOMATION","SMART_HOME_AUTOMATION","SMART_HOME_AUTOMATION","SMART_HOME_AUTOMATION","SMART_HOME_AUTOMATION","SMART_HOME_AUTOMATION","SMART_HOME_AUTOMATION","SMART_HOME_AUTOMATION","SMART_HOME_AUTOMATION","SMART_HOME_AUTOMATION","SMART_HOME_AUTOMATION","SMART_HOME_AUTOMATION","SMART_HOME_AUTOMATION","SMART_HOME_AUTOMATION","SMART_HOME_AUTOMATION",
                "WORKPLACE_ENERGY_SAVING","WORKPLACE_ENERGY_SAVING","WORKPLACE_ENERGY_SAVING","WORKPLACE_ENERGY_SAVING","WORKPLACE_ENERGY_SAVING","WORKPLACE_ENERGY_SAVING","WORKPLACE_ENERGY_SAVING","WORKPLACE_ENERGY_SAVING","WORKPLACE_ENERGY_SAVING","WORKPLACE_ENERGY_SAVING","WORKPLACE_ENERGY_SAVING","WORKPLACE_ENERGY_SAVING","WORKPLACE_ENERGY_SAVING","WORKPLACE_ENERGY_SAVING","WORKPLACE_ENERGY_SAVING","WORKPLACE_ENERGY_SAVING","WORKPLACE_ENERGY_SAVING","WORKPLACE_ENERGY_SAVING","WORKPLACE_ENERGY_SAVING","WORKPLACE_ENERGY_SAVING",
                "ENERGY_SAVING_TECHNOLOGY","ENERGY_SAVING_TECHNOLOGY","ENERGY_SAVING_TECHNOLOGY","ENERGY_SAVING_TECHNOLOGY","ENERGY_SAVING_TECHNOLOGY","ENERGY_SAVING_TECHNOLOGY","ENERGY_SAVING_TECHNOLOGY","ENERGY_SAVING_TECHNOLOGY","ENERGY_SAVING_TECHNOLOGY","ENERGY_SAVING_TECHNOLOGY","ENERGY_SAVING_TECHNOLOGY","ENERGY_SAVING_TECHNOLOGY","ENERGY_SAVING_TECHNOLOGY","ENERGY_SAVING_TECHNOLOGY","ENERGY_SAVING_TECHNOLOGY","ENERGY_SAVING_TECHNOLOGY","ENERGY_SAVING_TECHNOLOGY","ENERGY_SAVING_TECHNOLOGY","ENERGY_SAVING_TECHNOLOGY","ENERGY_SAVING_TECHNOLOGY",
                "ENERGY_SAVING_CHECKLIST","ENERGY_SAVING_CHECKLIST","ENERGY_SAVING_CHECKLIST","ENERGY_SAVING_CHECKLIST","ENERGY_SAVING_CHECKLIST","ENERGY_SAVING_CHECKLIST","ENERGY_SAVING_CHECKLIST","ENERGY_SAVING_CHECKLIST","ENERGY_SAVING_CHECKLIST","ENERGY_SAVING_CHECKLIST","ENERGY_SAVING_CHECKLIST","ENERGY_SAVING_CHECKLIST","ENERGY_SAVING_CHECKLIST","ENERGY_SAVING_CHECKLIST","ENERGY_SAVING_CHECKLIST","ENERGY_SAVING_CHECKLIST","ENERGY_SAVING_CHECKLIST","ENERGY_SAVING_CHECKLIST","ENERGY_SAVING_CHECKLIST","ENERGY_SAVING_CHECKLIST",
                "ENERGY_SAVING_COSTS","ENERGY_SAVING_COSTS","ENERGY_SAVING_COSTS","ENERGY_SAVING_COSTS","ENERGY_SAVING_COSTS","ENERGY_SAVING_COSTS","ENERGY_SAVING_COSTS","ENERGY_SAVING_COSTS","ENERGY_SAVING_COSTS","ENERGY_SAVING_COSTS","ENERGY_SAVING_COSTS","ENERGY_SAVING_COSTS","ENERGY_SAVING_COSTS","ENERGY_SAVING_COSTS","ENERGY_SAVING_COSTS","ENERGY_SAVING_COSTS","ENERGY_SAVING_COSTS","ENERGY_SAVING_COSTS","ENERGY_SAVING_COSTS","ENERGY_SAVING_COSTS",
                "ENERGY_SAVING_FINANCE","ENERGY_SAVING_FINANCE","ENERGY_SAVING_FINANCE","ENERGY_SAVING_FINANCE","ENERGY_SAVING_FINANCE","ENERGY_SAVING_FINANCE","ENERGY_SAVING_FINANCE","ENERGY_SAVING_FINANCE","ENERGY_SAVING_FINANCE","ENERGY_SAVING_FINANCE","ENERGY_SAVING_FINANCE","ENERGY_SAVING_FINANCE","ENERGY_SAVING_FINANCE","ENERGY_SAVING_FINANCE","ENERGY_SAVING_FINANCE","ENERGY_SAVING_FINANCE","ENERGY_SAVING_FINANCE","ENERGY_SAVING_FINANCE","ENERGY_SAVING_FINANCE","ENERGY_SAVING_FINANCE",
                "ENERGY_SAVING_GOALS","ENERGY_SAVING_GOALS","ENERGY_SAVING_GOALS","ENERGY_SAVING_GOALS","ENERGY_SAVING_GOALS","ENERGY_SAVING_GOALS","ENERGY_SAVING_GOALS","ENERGY_SAVING_GOALS","ENERGY_SAVING_GOALS","ENERGY_SAVING_GOALS","ENERGY_SAVING_GOALS","ENERGY_SAVING_GOALS","ENERGY_SAVING_GOALS","ENERGY_SAVING_GOALS","ENERGY_SAVING_GOALS","ENERGY_SAVING_GOALS","ENERGY_SAVING_GOALS","ENERGY_SAVING_GOALS","ENERGY_SAVING_GOALS","ENERGY_SAVING_GOALS",
                ]}

  def intentText(self,intentText):
        
    df = pd.DataFrame(self.train_data)
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(df['input'])


    clf = SVC()
    clf.fit(X_train, df['intent'])

    X_test = vectorizer.transform([intentText])
    predicted_intent = clf.predict(X_test)

    # Calculate accuracy on training data
    y_pred_train = clf.predict(X_train)
    accuracy_train = accuracy_score(df['intent'], y_pred_train)

    print("Predicted intent:", predicted_intent)
    print("Training accuracy:", accuracy_train)

    return str(predicted_intent[0])


    # clf = SVC()
    # clf.fit(X_train, df['intent'])
    # X_test = vectorizer.transform([intentText])

    # predicted_intent = clf.predict(X_test)
    # print("Predicted intent:", predicted_intent)
    # return  str(predicted_intent[0]) 

