#----------------------------------------------------------------------------------------------------#
#-------------------------------------------Data Clearning-------------------------------------------#
#----------------------------------------------------------------------------------------------------#
library(randomForest)
library(rpart)
library(rattle)

column_names <- function(){
  # Function extracts column names from names file and cleans them up
  col_names = readLines('census-just-names.txt')
  col_names = sapply(col_names, function(s){
    s = gsub("^\\s?\\|\\s?", "", s)
    s = strsplit(s, "\\t+")
    s = unlist(s)
  })
  col_names = as.data.frame(t(col_names))
  row.names(col_names) = NULL
  names(col_names) = c("variable", "abbrev")
  col_names$variable = as.character(col_names$variable)
  col_names$abbrev = as.character(col_names$abbrev)
  col_names = rbind(col_names, c("year", "YR"))
  col_names = rbind(col_names, c("target", "TARGET"))
  col_names = subset(col_names, !(variable %in% c("adjusted gross income", "federal income tax liability",
                        "total person income", "taxable income amount", "total person earnings")))
  
  col_names[col_names$variable=='mace',1] = 'race'
  return(col_names)
}

# Read in census training and test data and apply column names
col_names <- column_names()
census = read.csv('census-income.data',
                  strip.white=TRUE, na.strings=c("Not in universe", "?"), header=FALSE)
names(census) = col_names$variable
censusTest = read.csv('census-income.test',
                  strip.white=TRUE, na.strings=c("Not in universe", "?"), header=FALSE)
names(censusTest) = col_names$variable

# Impute missing values with median or most common
censusImputed = na.roughfix(census)
censusTestImputed = na.roughfix(censusTest)

numericVar = c('age', 'wage per hour', 'capital gains', 'capital losses', 'divdends from stocks',
               'instance weight', 'num persons worked for employer', 'weeks worked in year')

newCensusData = rbind(censusImputed, censusTestImputed)

censusImputedNumeric = newCensusData
for(name in names(censusImputedNumeric)){
  if(!(name %in% c(numericVar,"target"))){
    censusImputedNumeric[,name] = as.numeric(censusImputedNumeric[,name])
  }
}

train_end = length(censusImputed[,1])
test_end = length(censusTestImputed[,1])
numericCensus = censusImputedNumeric[1:train_end, ]
numericCensusTest = censusImputedNumeric[(train_end+1):(train_end + test_end), ]

write.csv(numericCensus, file='censusImputedNumeric.csv', row.names=F)
write.csv(numericCensusTest, file='censusTestImputedNumeric.csv', row.names=F)




#----------------------------------------------------------------------------------------------------#
#------------------------------------------------EDA-------------------------------------------------#
#----------------------------------------------------------------------------------------------------#

numericVarNames = c("Age", "Wage per Hour", "Capital Gains",
                    "Capital Losses", "Dividends from Stocks", 
                    "Instance Weight", 
                    "Number of Persons Worked for Employer", 
                    "Weeks Worked in a Year")
par(mfrow=c(8,1), oma=c(.5,.5,.5,.5), mar=c(1.5,1.5,1.5,1.5))
for(i in 1:length(numericVar)){
  x = census[,numericVar[i]]
  boxplot(x, main=numericVarNames[i], 
          horizontal=TRUE, col="lightblue", xaxt='n')
  x_loc = seq(min(x), max(x), length.out=5)
  axis(1, at=x_loc, tck=-0.1, labels=FALSE)
  text(x_loc, par("usr")[3]-.3, 
       labels=as.character(x_loc), xpd=TRUE)
}

categoricalVar = names(census)[!(names(census) %in% numericVar)]
categoricalVarNames = c("Class of Worker",
                        "Industry Code",
                        "Occupation Code",
                        "Education Level",
                        "Enrolled in Educational Institution Last Week",
                        "Marital Status",
                        "Major Industry Code",
                        "Major Occupation Code",
                        "Race",
                        "Hispanic Origin",
                        "Sex",
                        "Member of a Labor Union",
                        "Reason for Unemployment",
                        "Full or Part Time Employment Status",
                        "Tax Filer Status",
                        "Region of Previous Residence",
                        "State of Previous Residence",
                        "Detailed Household/Family Status",
                        "Detailed Household Summary in Household",
                        "Migration Code - Change in MSA",
                        "Migration Code - Change in Region",
                        "Migration Code - Move within Region",
                        "Live in this House 1 Year Ago",
                        "Migration Previous Residence in Sunbelt",
                        "Family Members Under 18",
                        "Country of Birth - Father",
                        "Country of Birth - Mother",
                        "Country of Birth - Self",
                        "Citizenship",
                        "Own Business or Self Employed",
                        "Fill Inc Questionnaire for Veteran's Admin",
                        "Veteran's Benefits",
                        "Year",
                        "Target")

par(mfrow=c(4,1), mar=c(2,18,2,2))
for(i in 1:length(categoricalVar)){
  table = sort(table(census[,categoricalVar[i]], useNA=TRUE), decreasing=TRUE)
  numCat = length(table)
  if(length(table)>5){
    other = sum(table[6:numCat])
    table = table[1:4]
    table[5] = other
    names(table)[5] = "Other"
  }
  barplot(rev(table), horiz=TRUE, main=categoricalVarNames[i], 
          las=2, xaxt='n', col="lightblue")
  x_loc = seq(0, max(table), length.out=5)
  axis(1, at=x_loc, tck=0.00, labels=FALSE)
  text(x_loc, par("usr")[3]-0.5, 
       labels=as.character(x_loc), xpd=TRUE)
}
