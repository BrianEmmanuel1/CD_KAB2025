from keras import models, callbacks, optimizers


class Model():
    
    def __init__(self, model : models.Model):
        self.keras_model = model
        #set optimizer
        self.optimizer = optimizers.Adam(learning_rate=0.001)
        #set a earlystopping to avoid overfitting
        self.callback = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    
    def compile(self):
        self.keras_model.compile(self.optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    
    def fit(self, train_data, validation_data, epochs):
        history = self.keras_model.fit(train_data, validation_data=validation_data, epochs=epochs, callbacks=[self.callback]) 
        return history
    
    def evaluate(self, test_data, output_data) -> None:
        print("\nEvaluation stage")
        evaluation = self.keras_model.evaluate(test_data, output_data)
        print(f"\nTest loss: {evaluation[0]:.4f}")
        print(f"Test accuracy: {evaluation[1]*100:.2f}%\n")
        
        
    #evaluate a set of values with the current model and print the predicted and real values
    
    def test_examples(self, test_data, output_data):
        print("Testing examples")
        
        prediction = self.keras_model.predict(test_data)
        
        #get the value where the activation was higher
        pred_idx = prediction.argmax(1)
        real_idx = output_data.argmax(1)
        
        print(f"predicted:   {pred_idx}")
        print(f"real output: {real_idx}")
    
    def set_learning_rate(self, learning_rate):
        self.optimizer.learning_rate = learning_rate
        
    def set_patience(self, patience):
        self.callback.patience = patience