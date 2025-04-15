import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import os
import sys
from simple_recommender import SimpleMarketingRecommender

class MarketingRecommenderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Digital Marketing Strategy Recommender for SMEs")
        self.root.geometry("900x700")
        self.root.configure(bg="#f5f5f5")
        
        # Initialize recommender
        self.recommender = SimpleMarketingRecommender()
        if not self.recommender.strategies:
            messagebox.showerror("Error", "Failed to load marketing strategies data.")
            self.root.destroy()
            return
            
        # Extract industries
        self.all_industries = self.extract_all_industries()
        
        # Create UI elements
        self.create_widgets()
        
    def extract_all_industries(self):
        """Extract all unique industries from the dataset"""
        all_industries = set()
        for strategy in self.recommender.strategies:
            industries = strategy['best_for_industry'].replace('"', '').split(',')
            all_industries.update(industries)
        return sorted(list(all_industries))
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(
            header_frame, 
            text="Digital Marketing Strategy Recommender for SMEs",
            font=("Arial", 16, "bold")
        ).pack()
        
        ttk.Label(
            header_frame,
            text="Find the most suitable marketing strategies based on your business needs",
            font=("Arial", 10)
        ).pack(pady=5)
        
        # Create input form
        form_frame = ttk.LabelFrame(main_frame, text="Your Business Profile", padding=10)
        form_frame.pack(fill=tk.X, pady=10)
        
        # Industry selection
        ttk.Label(form_frame, text="Industry:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.industry_var = tk.StringVar()
        industry_combo = ttk.Combobox(form_frame, textvariable=self.industry_var, width=30)
        industry_combo['values'] = self.all_industries
        industry_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
        industry_combo.current(0) if self.all_industries else None
        
        # Budget Level
        ttk.Label(form_frame, text="Budget Level:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.budget_var = tk.IntVar(value=3)
        budget_frame = ttk.Frame(form_frame)
        budget_frame.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(budget_frame, text="1").pack(side=tk.LEFT)
        ttk.Scale(
            budget_frame, 
            from_=1, 
            to=5, 
            orient=tk.HORIZONTAL, 
            variable=self.budget_var,
            length=200
        ).pack(side=tk.LEFT, padx=5)
        ttk.Label(budget_frame, text="5").pack(side=tk.LEFT)
        ttk.Label(budget_frame, text="(Very Low - Very High)").pack(side=tk.LEFT, padx=10)
        
        # Technical Skills
        ttk.Label(form_frame, text="Technical Skills:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.tech_var = tk.IntVar(value=3)
        tech_frame = ttk.Frame(form_frame)
        tech_frame.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(tech_frame, text="1").pack(side=tk.LEFT)
        ttk.Scale(
            tech_frame, 
            from_=1, 
            to=5, 
            orient=tk.HORIZONTAL, 
            variable=self.tech_var,
            length=200
        ).pack(side=tk.LEFT, padx=5)
        ttk.Label(tech_frame, text="5").pack(side=tk.LEFT)
        ttk.Label(tech_frame, text="(Beginner - Expert)").pack(side=tk.LEFT, padx=10)
        
        # Time Available
        ttk.Label(form_frame, text="Time Available:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.time_var = tk.IntVar(value=3)
        time_frame = ttk.Frame(form_frame)
        time_frame.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(time_frame, text="1").pack(side=tk.LEFT)
        ttk.Scale(
            time_frame, 
            from_=1, 
            to=5, 
            orient=tk.HORIZONTAL, 
            variable=self.time_var,
            length=200
        ).pack(side=tk.LEFT, padx=5)
        ttk.Label(time_frame, text="5").pack(side=tk.LEFT)
        ttk.Label(time_frame, text="(Very Limited - Abundant)").pack(side=tk.LEFT, padx=10)
        
        # Audience Size
        ttk.Label(form_frame, text="Target Audience Size:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.audience_var = tk.IntVar(value=3)
        audience_frame = ttk.Frame(form_frame)
        audience_frame.grid(row=4, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(audience_frame, text="1").pack(side=tk.LEFT)
        ttk.Scale(
            audience_frame, 
            from_=1, 
            to=5, 
            orient=tk.HORIZONTAL, 
            variable=self.audience_var,
            length=200
        ).pack(side=tk.LEFT, padx=5)
        ttk.Label(audience_frame, text="5").pack(side=tk.LEFT)
        ttk.Label(audience_frame, text="(Very Small - Very Large)").pack(side=tk.LEFT, padx=10)
        
        # Marketing Goal
        ttk.Label(form_frame, text="Primary Marketing Goal:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.goal_var = tk.StringVar(value="awareness")
        goal_frame = ttk.Frame(form_frame)
        goal_frame.grid(row=5, column=1, sticky=tk.W, pady=5)
        
        goals = [
            ("Increase Conversion Rates", "conversion"),
            ("Boost Brand Awareness", "awareness"),
            ("Generate More Leads", "leads"),
            ("Improve Customer Retention", "retention")
        ]
        
        for i, (text, value) in enumerate(goals):
            ttk.Radiobutton(
                goal_frame, 
                text=text, 
                variable=self.goal_var, 
                value=value
            ).grid(row=0, column=i, padx=5)
        
        # Generate Recommendations button
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            button_frame,
            text="Generate Recommendations",
            command=self.generate_recommendations,
            style="Accent.TButton"
        ).pack(pady=10)
        
        # Results area
        self.results_frame = ttk.LabelFrame(main_frame, text="Recommended Strategies", padding=10)
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Initial message
        self.results_text = scrolledtext.ScrolledText(
            self.results_frame, 
            wrap=tk.WORD, 
            width=80, 
            height=20,
            font=("Courier New", 10)
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        self.results_text.insert(tk.END, "Complete the form and click 'Generate Recommendations' to get personalized marketing strategies.")
        self.results_text.config(state=tk.DISABLED)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Configure style for buttons
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Arial", 11))
        
    def generate_recommendations(self):
        """Generate and display recommendations based on user inputs"""
        try:
            # Clear previous results
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            
            # Update status
            self.status_var.set("Generating recommendations...")
            self.root.update_idletasks()
            
            # Get user inputs
            industry = self.industry_var.get()
            budget_level = self.budget_var.get()
            technical_skill = self.tech_var.get()
            time_available = self.time_var.get()
            audience_size = self.audience_var.get()
            goal = self.goal_var.get()
            
            # Validate industry selection
            if not industry:
                messagebox.showwarning("Input Error", "Please select an industry.")
                self.status_var.set("Ready")
                return
            
            # Create user preferences dictionary
            user_prefs = {
                'budget_level': budget_level,
                'technical_skill': technical_skill,
                'time_available': time_available,
                'goal': goal,
                'industry': industry,
                'audience_size': audience_size
            }
            
            # Get recommendations
            recommendations = self.recommender.get_recommendations(user_prefs, top_n=3)
            
            if not recommendations:
                self.results_text.insert(tk.END, "No matching strategies found. Try adjusting your criteria.")
                self.status_var.set("No matches found")
                return
            
            # Display recommendations
            self.results_text.insert(tk.END, "TOP RECOMMENDED MARKETING STRATEGIES\n")
            self.results_text.insert(tk.END, "=" * 60 + "\n\n")
            
            for i, (strategy, similarity) in enumerate(recommendations, 1):
                self.results_text.insert(tk.END, f"RECOMMENDATION #{i} - Match Score: {similarity:.2f}\n")
                self.results_text.insert(tk.END, "-" * 60 + "\n")
                self.results_text.insert(tk.END, f"Strategy: {strategy['strategy_name']}\n")
                self.results_text.insert(tk.END, f"Best for: {strategy['best_for_industry']}\n\n")
                
                self.results_text.insert(tk.END, "Resource Requirements:\n")
                self.results_text.insert(tk.END, f"  Budget required: {'$' * strategy['budget_required']} ({strategy['budget_required']}/5)\n")
                self.results_text.insert(tk.END, f"  Technical expertise: {'*' * strategy['technical_expertise']} ({strategy['technical_expertise']}/5)\n")
                self.results_text.insert(tk.END, f"  Time investment: {'⏱' * strategy['time_investment']} ({strategy['time_investment']}/5)\n\n")
                
                self.results_text.insert(tk.END, "Expected Outcomes:\n")
                self.results_text.insert(tk.END, f"  Conversion Rate: {'↑' * strategy['conversion_rate']} ({strategy['conversion_rate']}/5)\n")
                self.results_text.insert(tk.END, f"  Brand Awareness: {'👁' * strategy['brand_awareness']} ({strategy['brand_awareness']}/5)\n")
                self.results_text.insert(tk.END, f"  Lead Generation: {'⚡' * strategy['lead_generation']} ({strategy['lead_generation']}/5)\n")
                self.results_text.insert(tk.END, f"  Customer Retention: {'♥' * strategy['customer_retention']} ({strategy['customer_retention']}/5)\n")
                self.results_text.insert(tk.END, "\n" + "-" * 60 + "\n\n")
            
            # Update status
            self.status_var.set(f"Generated {len(recommendations)} recommendations")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set("Error occurred")
        finally:
            self.results_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = MarketingRecommenderGUI(root)
    root.mainloop() 