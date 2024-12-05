class AUK:
    def __init__(self, probabilities, labels, integral='trapezoid'):
        self.probabilities = probabilities
        self.labels = labels
        self.integral = integral
        if integral not in ['trapezoid','max','min']:
            raise ValueError('"'+str(integral)+'"'+ ' is not a valid integral value. Choose between "trapezoid", "min" or "max"')
        self.probabilities_set = sorted(list(set(probabilities)))
    
    #make predictions based on the threshold value and self.probabilities
    def _make_predictions(self, threshold):
        predictions = []
        for prob in self.probabilities:
            if prob >= threshold:
                predictions.append(1)
            else: 
                predictions.append(0)
        return predictions
    
    #make list with kappa scores for each threshold
    def kappa_curve(self):
        kappa_list = []
        
        for thres in self.probabilities_set:
            preds = self._make_predictions(thres)
            tp, tn, fp, fn = self.confusion_matrix(preds)
            k = self.calculate_kappa(tp, tn, fp, fn)
            kappa_list.append(k)
        return self._add_zero_to_curve(kappa_list)
    
    #make list with fpr scores for each threshold
    def fpr_curve(self):
        fpr_list = []
        
        for thres in self.probabilities_set:
            preds = self._make_predictions(thres)
            tp, tn, fp, fn = self.confusion_matrix(preds)
            fpr = self.calculate_fpr(fp, tn)
            fpr_list.append(fpr)
        return self._add_zero_to_curve(fpr_list)
    
   
    #calculate confusion matrix
    def confusion_matrix(self, predictions):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i, pred in enumerate(predictions):
            if pred == self.labels[i]:
                if pred == 1:
                    tp += 1
                else: 
                    tn += 1
            elif pred == 1:
                fp += 1
            else: fn += 1
            tot = tp + tn + fp + fn
        return tp/tot, tn/tot, fp/tot, fn/tot
        
    #Calculate AUK
    def calculate_auk(self):        
        auk=0
        fpr_list = self.fpr_curve()
            
        for i, prob in enumerate(self.probabilities_set[:-1]):
            x_dist = abs(fpr_list[i+1] - fpr_list[i])
                
            preds = self._make_predictions(prob) 
            tp, tn, fp, fn = self.confusion_matrix(preds)
            kapp1 = self.calculate_kappa(tp, tn, fp, fn)
                
            preds = self._make_predictions(self.probabilities_set[i+1]) 
            tp, tn, fp, fn = self.confusion_matrix(preds)
            kapp2 = self.calculate_kappa(tp, tn, fp, fn)
                
            y_dist = abs(kapp2-kapp1)
            bottom = min(kapp1, kapp2)*x_dist
            auk += bottom
            if self.integral == 'trapezoid':
                top = (y_dist * x_dist)/2
                auk += top
            elif self.integral == 'max':
                top = (y_dist * x_dist)
                auk += top
            else:
                continue
        return auk
    
    #Calculate the false-positive rate
    def calculate_fpr(self, fp, tn):
        return fp/(fp+tn)
    
    #Calculate kappa score
    def calculate_kappa(self, tp, tn, fp, fn):
        acc = tp + tn
        p = tp + fn
        p_hat = tp + fp
        n = fp + tn
        n_hat = fn + tn
        p_c = p * p_hat + n * n_hat
        return (acc - p_c) / (1 - p_c)
    
    #Add zero to appropriate position in list
    def _add_zero_to_curve(self, curve):
        min_index = curve.index(min(curve)) 
        if min_index> 0:
            curve.append(0)
        else: curve.insert(0,0)
        return curve
    #Add zero to appropriate position in list
    def _add_zero_to_curve(self, curve):
        min_index = curve.index(min(curve)) 
        if min_index> 0:
            curve.append(0)
        else: curve.insert(0,0)
        return curve